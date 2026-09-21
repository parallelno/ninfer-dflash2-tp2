#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

DFlashFeatureSink make_dflash_prefill_sink(PrefillContext& state) {
    if (!state.execution.io.dflash_decode || state.dflash_host_ingress == nullptr) {
        throw std::logic_error("DFlash prefill controls are unavailable");
    }
    return dflash_feature_sink(
        state, [&state](const Tensor& features, const Tensor& positions, bool rewrite_checkpoint) {
            auto& frame  = *state.execution.io.dflash_decode;
            Tensor count = frame.append_counts.slice(0, 0, 1);
            Tensor lane  = frame.state_destination_slots.slice(0, 0, 1);
            Tensor row   = frame.dflash_kv_table_rows.slice(0, 0, 1);
            ops::set_i32_scalar(count, features.ne[1], state.execution.device.stream);
            const auto exact = static_cast<std::uint32_t>(features.ne[1]);
            dflash_append_context(state, features, positions, count, lane, row, {exact, exact});
            (void)rewrite_checkpoint;
        });
}

} // namespace

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t state_source_slot,
                         std::int32_t state_destination_slot, std::uint32_t mtp_proposal_extent) {
    card.set_sampling(sampling);
    card.set_linear_state_slots(state_source_slot, state_destination_slot);
    card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    card.set_mtp_proposal_extent(mtp_proposal_extent);
    if (execution.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_ids() == nullptr ||
        card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

PrefillChunkResult prefill_text_chunk(PrefillContext& state, std::span<const TokenId> ids,
                                      std::uint32_t nominal_length,
                                      std::optional<std::uint32_t> split_frontier,
                                      bool finalize_at_end) {
    std::optional<TpExecution> tp = tp_execution(state.execution);
    if (tp) { tp->mtp_kv = state.mtp_kv_peer; }
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                     tp ? &*tp : nullptr);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    const std::span<const int> prompt(ids.data(), ids.size());
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                  sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end);
}

// Wavefront entry (prototype, H18 in fork/research-prefill-improvements2.md): hands ALL of the
// remaining prompt to TextContext::prefill_wavefront_tp2, which pipelines consecutive chunks at
// layer granularity. Returns processed_tokens == 0 when the wavefront does not apply (tp1, MTP
// prompt prep, split frontier, single chunk, arena pressure); the caller then runs the ordinary
// per-chunk path. With DFlash active each slot gets its own feature sink (slot 1's capture
// buffers are cloned inside the wavefront call).
PrefillChunkResult prefill_text_wavefront(PrefillContext& state, std::span<const TokenId> ids,
                                          std::uint32_t nominal_length, bool finalize_at_end,
                                          std::span<const std::uint32_t> split_frontiers) {
    std::optional<TpExecution> tp = tp_execution(state.execution);
    if (!tp) { return PrefillChunkResult{}; }
    tp->mtp_kv = state.mtp_kv_peer;
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                     &*tp);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    const std::span<const int> prompt(ids.data(), ids.size());
    DFlashFeatureSink sink0;
    DFlashFeatureSink sink1;
    DFlashFeatureSink* sinks[2] = {nullptr, nullptr};
    if (state.dflash != nullptr) {
        sink0    = make_dflash_prefill_sink(state);
        sink1    = make_dflash_prefill_sink(state);
        sinks[0] = &sink0;
        sinks[1] = &sink1;
    }
    const std::uint32_t processed = card.prefill_chunk_wavefront(
        prompt, state.text_kv_base, nominal_length, finalize_at_end, sinks, split_frontiers);
    return PrefillChunkResult{.processed_tokens = processed,
                              .finalized = finalize_at_end && processed == nominal_length,
                              .timing    = {}};
}

PrefillChunkResult prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                                            VisionPrefillSession& vision,
                                            std::uint32_t nominal_length,
                                            std::optional<std::uint32_t> split_frontier,
                                            bool finalize_at_end) {
    std::optional<TpExecution> tp = tp_execution(state.execution);
    if (tp) { tp->mtp_kv = state.mtp_kv_peer; }
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                     tp ? &*tp : nullptr);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision,
                                  finalize_at_end, sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision, finalize_at_end);
}

void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge) {
    if (!state.mtp_kv.valid() || bridge.previous_hidden == nullptr || state.text_kv_base == 0 ||
        bridge.position < 0 ||
        static_cast<std::uint32_t>(bridge.position) + 1 != state.text_kv_base) {
        throw std::logic_error("multimodal MTP bridge does not match the reusable frontier");
    }

    Tensor bridge_token = state.execution.io.mtp->target_input_ids.slice(0, 0, 1);
    const TokenId token = prompt.token_ids[state.text_kv_base];
    CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               state.execution.device.stream));

    Tensor visual_embedding;
    const Tensor* composed_embedding = nullptr;
    if (prompt.token_types[state.text_kv_base] != 0) {
        const VisionChunk chunk = vision.prepare_chunk(state.text_kv_base, 1);
        if (chunk.control == nullptr) {
            throw std::logic_error("visual MTP bridge has no encoded Vision item");
        }
        const auto& scatter = chunk.control->scatter_indices;
        const auto column   = std::lower_bound(scatter.begin(), scatter.end(),
                                               static_cast<std::int32_t>(state.text_kv_base));
        if (column == scatter.end() || *column != static_cast<std::int32_t>(state.text_kv_base) ||
            static_cast<std::uint8_t>(chunk.control->modality) !=
                prompt.token_types[state.text_kv_base]) {
            throw std::logic_error("visual MTP bridge does not match Vision scatter metadata");
        }
        visual_embedding =
            chunk.embeddings.slice(1, static_cast<std::int32_t>(column - scatter.begin()), 1);
        composed_embedding = &visual_embedding;
    }

    mtp_bridge_and_propose(state, bridge_token, *bridge.previous_hidden, bridge.position,
                           bridge.rope_position, false, composed_embedding);
}

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] != 1 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [hidden,1]");
    }
    if (state.execution.peer != nullptr) {
        // UNREACHABLE BACKSTOP. The output head is vocabulary-split at tp2, so this bonus-token
        // path cannot run rank 0's whole-head GEMM. It is reached only by a zero-suffix reuse
        // plan, and `plan_request_for_lane` downgrades those to a full reset at tp2 precisely so
        // this cannot happen during execution -- an exception here would take the executor down
        // with it. Kept so a future planner change cannot reintroduce the case silently.
        throw std::logic_error(
            "tensor-parallel prefix reuse cannot sample from a restored hidden state yet");
    }
    state.execution.work.reset();
    Tensor logits = state.execution.io.logits.slice(1, 0, 1);
    ops::linear(hidden, state.execution.model.output_head, logits, state.execution.device.stream);
    CUDA_CHECK(cudaMemcpyAsync(state.execution.io.pos.data, &absolute_position,
                               sizeof(absolute_position), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    ops::sample(logits, state.execution.io.token, TextConfig::token_domain, state.sampling,
                state.execution.io.pos, purpose, state.execution.work,
                state.execution.device.stream);
    state.execution.work.reset();
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
