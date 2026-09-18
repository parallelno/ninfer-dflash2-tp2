#pragma once

#include "ninfer/targets/qwen3_6/prepared_prompt.h"
#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::targets::qwen3_6 {

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

struct StartupFeatures {
    bool vision                    = false;
    // tp rank (index into EngineOptions::devices) that holds the Vision tower; 0 at tp 1.
    int vision_rank = 0;
    // Per-item merged-token ceiling the Vision workspace and the media processor are bound to.
    std::uint32_t max_vision_item_tokens = static_cast<std::uint32_t>(kMaximumVisionItemTokens);
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept {
        return is_masked_draft_backend(speculative);
    }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }
};

// Rank of `options.vision_device` within the execution device list; 0 when unset or absent
// (registry validation rejects an absent id before anything is built from this value).
[[nodiscard]] inline int vision_rank_for(const EngineOptions& options) noexcept {
    if (!options.enable_vision || options.vision_device < 0) { return 0; }
    for (std::size_t rank = 0; rank < options.devices.size(); ++rank) {
        if (options.devices[rank] == options.vision_device) { return static_cast<int>(rank); }
    }
    return 0;
}

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    const std::uint32_t item_tokens =
        options.max_vision_tokens == 0
            ? static_cast<std::uint32_t>(kMaximumVisionItemTokens)
            : std::min<std::uint32_t>(options.max_vision_tokens,
                                      static_cast<std::uint32_t>(kMaximumVisionItemTokens));
    return StartupFeatures{
        .vision                 = options.enable_vision,
        .vision_rank            = vision_rank_for(options),
        .max_vision_item_tokens = options.enable_vision
                                      ? item_tokens
                                      : static_cast<std::uint32_t>(kMaximumVisionItemTokens),
        .speculative            = options.speculative.backend,
        .proposal_head          = options.speculative.proposal_head,
    };
}

} // namespace ninfer::targets::qwen3_6
