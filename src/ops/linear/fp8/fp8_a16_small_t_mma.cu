#include "ops/linear/fp8/fp8_launch.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a16_small_t_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <typename Geometry, int ActiveTokens>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Fp8VocabularyA16SmallTMmaProductionSchedule<ActiveTokens>::Type;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_a16_small_t_mma_kernel<Geometry, ActiveTokens, Schedule, Fp8ContiguousOutput, true>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_tile<Geometry, 8 * (static_cast<int>(Offsets) + 1)>...};
}

constexpr auto kFullLaunchers =
    make_launchers<Fp8VocabularyGeometry>(std::make_index_sequence<6>{});
constexpr auto kTp2Launchers =
    make_launchers<Fp8VocabularyTp2ColumnGeometry>(std::make_index_sequence<6>{});

} // namespace

void launch_fp8_vocabulary_a16_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    const bool full = weight.n == Fp8VocabularyGeometry::kOutputRows &&
                      weight.k == Fp8VocabularyGeometry::kInputRows;
    const bool tp2 = weight.n == Fp8VocabularyTp2ColumnGeometry::kOutputRows &&
                     weight.k == Fp8VocabularyTp2ColumnGeometry::kInputRows;
    if ((!full && !tp2) ||
        x.ne[1] < kFp8VocabularyFirstA16SmallTMmaT || x.ne[1] > kFp8VocabularyLastA16SmallTMmaT) {
        throw std::invalid_argument("fp8 vocabulary A16 small-T MMA: invalid exact problem");
    }
    const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
    (full ? kFullLaunchers : kTp2Launchers)[index](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
