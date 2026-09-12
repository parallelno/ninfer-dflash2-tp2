#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::CausalAttentionExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens);
    }
    if (frame.proposal_q.data != nullptr) {
        ops::speculative_accept_sparse_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            TextConfig::token_domain, frame.sampling, {false}, execution.work,
            execution.device.stream);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts, TextConfig::token_domain, frame.sampling, execution.work,
            execution.device.stream);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
}

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          TargetVerifyFrameView peer,
                          ops::CausalAttentionExecutionEnvelope envelope) {
    if (execution.peer == nullptr) {
        throw std::logic_error("tensor-parallel target verify requires a peer");
    }
    if (frame.replay_records == nullptr || peer.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    if (peer.feature_sink != nullptr) {
        throw std::logic_error("tensor-parallel peer target verify cannot capture features");
    }
    if (execution.peer->continuation_hidden_store == nullptr) {
        throw std::logic_error("tensor-parallel target verify has no peer continuation store");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    card.target_verify_batch({frame.ids, peer.ids},
                             {frame.cache_positions, peer.cache_positions},
                             {frame.rope_positions, peer.rope_positions},
                             {frame.valid_columns, peer.valid_columns},
                             {frame.kv_table_rows, peer.kv_table_rows},
                             {frame.state_source_slots, peer.state_source_slots},
                             envelope, {frame.target_hidden, peer.target_hidden},
                             {frame.target_logits, peer.target_logits},
                             {frame.target_tokens, peer.target_tokens}, frame.feature_sink);
    if (frame.proposal_q.data != nullptr) {
        ops::speculative_accept_sparse_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            TextConfig::token_domain, frame.sampling, {false}, execution.work,
            execution.device.stream);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts, TextConfig::token_domain, frame.sampling, execution.work,
            execution.device.stream);
    }
    const ops::PeerEvents& events = *execution.peer->events;
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), execution.device.stream));
    CUDA_CHECK(cudaSetDevice(execution.peer->device->device));
    CUDA_CHECK(cudaStreamWaitEvent(execution.peer->device->stream, events.inputs_ready(0), 0));
    CUDA_CHECK(cudaMemcpyAsync(peer.accepted_drafts.data, frame.accepted_drafts.data,
                               frame.accepted_drafts.bytes(), cudaMemcpyDeviceToDevice,
                               execution.peer->device->stream));
    ops::speculative_select_accepted_hidden(peer.target_hidden, peer.accepted_drafts,
                                            peer.selected_hidden,
                                            execution.peer->device->stream);
    CUDA_CHECK(cudaSetDevice(execution.device.device));
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
    ops::scatter(peer.selected_hidden, peer.state_destination_slots,
                 *execution.peer->continuation_hidden_store, execution.peer->device->stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
