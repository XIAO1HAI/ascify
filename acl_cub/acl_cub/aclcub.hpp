/*
 * acl_cub/aclcub.hpp  (0612 iteration-2: double-buffered + auto-fold AllReduce)
 * ----------------------------------------------------------------------------
 * Software re-implementation of the subset of cub used by oneflow's
 * oneflow/core/cuda/softmax.cuh. Resolves the post-ascify include
 *
 *     #include <acl_cub/aclcub.hpp>
 *
 * that ascify emits when rewriting `#include <cub/cub.cuh>`.
 *
 * Conversion contract (from the caller's point of view, ascify-driven):
 *     <cub/cub.cuh>        ->  <acl_cub/aclcub.hpp>
 *     cub::BlockReduce     ->  aclcub::BlockReduce
 *   ... and nothing else. oneflow's SumOp<T>/MaxOp<T> stay verbatim; this
 *   header recognizes them via partial specialization of OpTagOf at the
 *   bottom of the file ("Migration recognizer map list").
 *
 * Public surface:
 *   aclcub::Sum / aclcub::Max / aclcub::Min                (op tags)
 *   aclcub::BlockReduce<T, BLOCK>::TempStorage
 *   aclcub::BlockReduce<T, BLOCK>::Reduce(val, op)         cub-compatible; result
 *                                                          valid in thread 0 (in
 *                                                          fact across all of warp
 *                                                          0 — an aclcub superset of
 *                                                          cub's thread-0-only).
 *   aclcub::BlockReduce<T, BLOCK>::Sum(val)
 *   aclcub::BlockReduce<T, BLOCK>::AllReduce(val, op)      result in ALL threads,
 *                                                          ONE barrier (single-buf;
 *                                                          caller MUST keep its own
 *                                                          reuse-guard barrier).
 *   aclcub::BlockReduce<T, BLOCK>::AllReduce(val, op, iter) result in ALL threads,
 *                                                          ONE barrier, DOUBLE
 *                                                          BUFFERED by iteration
 *                                                          parity so a grid-stride
 *                                                          caller can DROP its
 *                                                          end-of-row reuse-guard
 *                                                          barrier. THE recommended
 *                                                          overload for softmax
 *                                                          BlockAllReduce.
 *
 *   *** iter CONTRACT (read this) ***  `iter` is NOT the global row index. It
 *   must increment by exactly 1 on each successive call that REUSES the same
 *   TempStorage within a block — i.e. the block-local grid-stride iteration
 *   count 0,1,2,... Its low bit selects the UB buffer, so it must strictly
 *   alternate per reuse. Passing the global row index (which advances by
 *   gridDim.x) is correct ONLY when gridDim.x is odd; with an even grid the
 *   parity is constant, the double buffer degenerates to one buffer, and the
 *   dropped reuse-guard barrier reintroduces a write-after-read race. So pass a
 *   per-iteration counter. (Proof of race-freedom under a strictly-alternating
 *   iter: same-buffer accesses are then >=2 iterations apart; the one publish
 *   barrier per iteration prevents any thread from advancing 2 iterations ahead
 *   of another — to reach iteration N+2's publish-write a thread must clear
 *   N+1's barrier, which requires every thread, including a lagging reader, to
 *   have finished iteration N's fold-read.)
 *
 * Both fast-path AllReduce overloads pick the second-level fold at COMPILE TIME
 * from the warp count (kWarps is constexpr; the choice is made by overload
 * resolution on a BoolConst tag, so the untaken fold is NOT instantiated):
 *   kWarps <= 4 : per-thread serial fold of the kWarps aggregates  (cheapest
 *                 for 2-4 of them; well-formed for any arithmetic T).
 *   kWarps >= 5 : one asc_reduce_* per warp                        (avoids the
 *                 serial-fold + UB-bank contention that dominates at 8/16/32
 *                 warps; ~3x faster than serial at block=1024). Requires T in
 *                 {float,int32_t,uint32_t} (the AscReduce overload set).
 * Measured optimal at every block size on 950PR-28CUBE (see
 * 0612_aclcub_process.md "Iteration-2 / crossover").
 *
 * Generic path (any non-{Sum,Max,Min} callable, OpTag=void): SMEM tree reduction
 * calling op(a, b) — log2(BLOCK) barriers, NOT one barrier, and NOT double
 * buffered (it uses its own per-call UB scratch, so reuse is safe, but the
 * single-barrier / drop-your-guard-barrier perf promise does NOT apply; the iter
 * argument is ignored). Preserves cub's "any binary functor" contract.
 */

#ifndef ACL_CUB_ACLCUB_HPP_
#define ACL_CUB_ACLCUB_HPP_

#include <cstdint>

// NOTE: no <limits> / <type_traits>. ccec's device pass (-x dpp,
// --cce-aicore-arch=dav-c310-vec) does not provide a working host STL.
#include <simt_api/device_warp_functions.h>
#include <simt_api/device_sync_functions.h>

namespace aclcub {

// ----------------------------------------------------------------------------
#ifndef ACLCUB_WARP_SIZE
#define ACLCUB_WARP_SIZE 32
#endif
constexpr int kWarpSize = ACLCUB_WARP_SIZE;

// ----------------------------------------------------------------------------
// Reduction op tags.
// ----------------------------------------------------------------------------
struct Sum {
  template <typename T>
  __aicore__ inline T operator()(const T& a, const T& b) const { return a + b; }
};
struct Max {
  template <typename T>
  __aicore__ inline T operator()(const T& a, const T& b) const { return a > b ? a : b; }
};
struct Min {
  template <typename T>
  __aicore__ inline T operator()(const T& a, const T& b) const { return a < b ? a : b; }
};

// ----------------------------------------------------------------------------
// Reduction identities (seed for padding lanes in the warp-primitive fold).
// Float uses +/-inf; integer types are specialized to the true extrema because
// casting +/-__builtin_inff() to an integer is implementation-defined.
// ----------------------------------------------------------------------------
template <typename T, typename OpTag> struct Identity;
template <typename T> struct Identity<T, Sum> {
  __aicore__ inline static T value() { return T(0); }
};
template <typename T> struct Identity<T, Max> {
  __aicore__ inline static T value() { return static_cast<T>(-__builtin_inff()); }
};
template <typename T> struct Identity<T, Min> {
  __aicore__ inline static T value() { return static_cast<T>(__builtin_inff()); }
};
// Integer extrema (full specializations; AscReduce supports int32_t/uint32_t).
template <> struct Identity<int32_t,  Max> { __aicore__ inline static int32_t  value() { return -2147483647 - 1; } };
template <> struct Identity<int32_t,  Min> { __aicore__ inline static int32_t  value() { return  2147483647; } };
template <> struct Identity<uint32_t, Max> { __aicore__ inline static uint32_t value() { return 0u; } };
template <> struct Identity<uint32_t, Min> { __aicore__ inline static uint32_t value() { return 4294967295u; } };

// Warp-wide AllReduce via the Ascend SIMT warp primitives (routed through
// non-template tag-typed forwarders to avoid two-phase-lookup binding issues).
__aicore__ inline float    AscReduce(float    v, Sum) { return asc_reduce_add(v); }
__aicore__ inline int32_t  AscReduce(int32_t  v, Sum) { return asc_reduce_add(v); }
__aicore__ inline uint32_t AscReduce(uint32_t v, Sum) { return asc_reduce_add(v); }
__aicore__ inline float    AscReduce(float    v, Max) { return asc_reduce_max(v); }
__aicore__ inline int32_t  AscReduce(int32_t  v, Max) { return asc_reduce_max(v); }
__aicore__ inline uint32_t AscReduce(uint32_t v, Max) { return asc_reduce_max(v); }
__aicore__ inline float    AscReduce(float    v, Min) { return asc_reduce_min(v); }
__aicore__ inline int32_t  AscReduce(int32_t  v, Min) { return asc_reduce_min(v); }
__aicore__ inline uint32_t AscReduce(uint32_t v, Min) { return asc_reduce_min(v); }

template <typename T> __aicore__ inline T WarpReduce(T v, Sum) { return AscReduce(v, Sum{}); }
template <typename T> __aicore__ inline T WarpReduce(T v, Max) { return AscReduce(v, Max{}); }
template <typename T> __aicore__ inline T WarpReduce(T v, Min) { return AscReduce(v, Min{}); }

// Scalar binary apply for a reduction tag — used by the serial second-level fold.
template <typename T> __aicore__ inline T TagApply(T a, T b, Sum) { return a + b; }
template <typename T> __aicore__ inline T TagApply(T a, T b, Max) { return a > b ? a : b; }
template <typename T> __aicore__ inline T TagApply(T a, T b, Min) { return a < b ? a : b; }

// Non-template forwarders for asc_shfl_xor.
__aicore__ inline float    AscShflXor(float    v, int mask, int width = kWarpSize) { return asc_shfl_xor(v, mask, width); }
__aicore__ inline int32_t  AscShflXor(int32_t  v, int mask, int width = kWarpSize) { return asc_shfl_xor(v, mask, width); }
__aicore__ inline uint32_t AscShflXor(uint32_t v, int mask, int width = kWarpSize) { return asc_shfl_xor(v, mask, width); }

// Hand-rolled std::conditional (no <type_traits> in device pass).
template <bool C, typename T, typename F> struct Cond { using type = T; };
template <typename T, typename F> struct Cond<false, T, F> { using type = F; };

// Compile-time mapping: arbitrary Op -> {Sum, Max, Min}, or `void`.
template <typename Op>
struct OpTagOf {
  using type =
      typename Cond<__is_base_of(Sum, Op), Sum,
      typename Cond<__is_base_of(Max, Op), Max,
      typename Cond<__is_base_of(Min, Op), Min,
                    void>::type>::type>::type;
};

template <typename Tag> struct TagHolder {};
template <bool B> struct BoolConst {};   // compile-time fold selector

// ----------------------------------------------------------------------------
enum BlockReduceAlgorithm {
  ACLCUB_BLOCK_REDUCE_ASC_REDUCE = 0,  // default: asc_reduce_* warp primitive
  ACLCUB_BLOCK_REDUCE_ASC_SHFL   = 1   // log2(warp) asc_shfl_xor butterfly, cub-style
};

// ----------------------------------------------------------------------------
// BlockReduce<T, BLOCK_DIM_X, ALGORITHM = ACLCUB_BLOCK_REDUCE_ASC_REDUCE>
// ----------------------------------------------------------------------------
template <typename T, int BLOCK_DIM_X,
          BlockReduceAlgorithm ALGORITHM = ACLCUB_BLOCK_REDUCE_ASC_REDUCE>
class BlockReduce {
 public:
  static constexpr int kBlockThreads = BLOCK_DIM_X;
  static constexpr int kWarpThreads  = kWarpSize;
  static constexpr int kWarps        = (kBlockThreads + kWarpThreads - 1) / kWarpThreads;
  static_assert(kWarps <= kWarpSize,
                "aclcub::BlockReduce: kWarps must fit in a single warp for the "
                "second-level reduction. Cap BLOCK_DIM_X at kWarpSize*kWarpSize.");

  // Double-buffered: [iter_parity][warp]. The single-barrier Reduce/AllReduce(.,.)
  // overloads use buffer 0; AllReduce(.,.,iter) alternates buffers by iter parity
  // (see iter CONTRACT at file top) so consecutive grid-stride iterations never
  // touch the same UB slot, letting the caller drop the end-of-row reuse-guard
  // barrier. Cost: 2*kWarps*sizeof(T) of UB (256 B at block=1024) — negligible.
  struct _TempStorage { T warp_aggregates[2][kWarps]; };
  struct TempStorage { _TempStorage data; };

  __aicore__ inline explicit BlockReduce(TempStorage& temp_storage)
      : temp_storage_(temp_storage.data),
        linear_tid_(static_cast<int>(threadIdx.x)),
        warp_id_(linear_tid_ / kWarpThreads),
        lane_id_(linear_tid_ % kWarpThreads) {}

  // cub-compatible: any functor accepted. Result valid in thread 0.
  template <typename Op>
  __aicore__ inline T Reduce(T input, Op op) {
    using OpTag = typename OpTagOf<Op>::type;
    return ReduceDispatch(input, op, TagHolder<OpTag>{});
  }
  __aicore__ inline T Sum(T input) {
    return ReduceDispatch(input, ::aclcub::Sum{}, TagHolder<::aclcub::Sum>{});
  }

  // AllReduce (single-buffer): result in EVERY thread, ONE barrier. The caller
  // must still guard TempStorage reuse across iterations with its own barrier.
  template <typename Op>
  __aicore__ inline T AllReduce(T input, Op op) {
    using OpTag = typename OpTagOf<Op>::type;
    return AllReduceDispatch(input, op, TagHolder<OpTag>{});
  }

  // AllReduce (double-buffered): result in EVERY thread, ONE barrier, race-free
  // reuse so the grid-stride caller can DROP its end-of-row reuse-guard barrier.
  // `iter` MUST be a per-block iteration counter incrementing by 1 on each reuse
  // of this TempStorage (NOT the global row — see iter CONTRACT at file top).
  // THE recommended overload for softmax BlockAllReduce (replaces
  // Reduce()+broadcast-write+__syncthreads(), 2 barriers, with a single barrier).
  template <typename Op>
  __aicore__ inline T AllReduce(T input, Op op, int iter) {
    using OpTag = typename OpTagOf<Op>::type;
    return AllReduceDB<0>(input, op, TagHolder<OpTag>{}, iter);
  }

  // Explicit fold-strategy overloads (diagnostic / benchmark; FOLD=1 serial,
  // FOLD=2 warp). The auto AllReduce(.,.,iter) above (FOLD=0) is preferred.
  template <typename Op>
  __aicore__ inline T AllReduceSerial(T input, Op op, int iter) {
    using OpTag = typename OpTagOf<Op>::type;
    return AllReduceDB<1>(input, op, TagHolder<OpTag>{}, iter);
  }
  template <typename Op>
  __aicore__ inline T AllReduceW(T input, Op op, int iter) {
    using OpTag = typename OpTagOf<Op>::type;
    return AllReduceDB<2>(input, op, TagHolder<OpTag>{}, iter);
  }

 private:
  // ----- Reduce (result in thread 0; in fact across all of warp 0) -----------
  template <typename Op, typename Tag>
  __aicore__ inline T ReduceDispatch(T input, Op /*op*/, TagHolder<Tag>) {
    if (kWarps == 1) { return ::aclcub::WarpReduce(input, Tag{}); }
    T warp_agg = ::aclcub::WarpReduce(input, Tag{});
    if (lane_id_ == 0) { temp_storage_.warp_aggregates[0][warp_id_] = warp_agg; }
    asc_syncthreads();
    if (warp_id_ == 0) {
      T v = (lane_id_ < kWarps) ? temp_storage_.warp_aggregates[0][lane_id_]
                                : Identity<T, Tag>::value();
      warp_agg = ::aclcub::WarpReduce(v, Tag{});
    }
    return warp_agg;  // valid in thread 0 (and across warp 0)
  }

  // ----- Second-level folds of the kWarps published aggregates (buffer p) ----
  template <typename Tag>
  __aicore__ inline T FoldSerial(int p, TagHolder<Tag>) {
    T acc = temp_storage_.warp_aggregates[p][0];
#pragma unroll
    for (int i = 1; i < kWarps; ++i)
      acc = ::aclcub::TagApply(acc, temp_storage_.warp_aggregates[p][i], Tag{});
    return acc;
  }
  template <typename Tag>
  __aicore__ inline T FoldWarp(int p, TagHolder<Tag>) {
    T v = (lane_id_ < kWarps) ? temp_storage_.warp_aggregates[p][lane_id_]
                              : Identity<T, Tag>::value();
    return ::aclcub::WarpReduce(v, Tag{});
  }
  // Compile-time fold selection via overload resolution on BoolConst, so the
  // UNTAKEN fold is never instantiated (FoldWarp needs the {float,int32,uint32}
  // AscReduce overloads; this keeps BlockReduce<otherT,smallblock> compilable
  // since only FoldSerial — generic over T — is instantiated when kWarps<=4).
  template <typename Tag>
  __aicore__ inline T FoldPick(int p, TagHolder<Tag> h, BoolConst<true>)  { return FoldSerial(p, h); }
  template <typename Tag>
  __aicore__ inline T FoldPick(int p, TagHolder<Tag> h, BoolConst<false>) { return FoldWarp(p, h); }
  template <typename Tag>
  __aicore__ inline T FoldAuto(int p, TagHolder<Tag> h) {
    return FoldPick(p, h, BoolConst<(kWarps <= 4)>{});
  }

  // ----- AllReduce single-buffer (auto fold) --------------------------------
  template <typename Op, typename Tag>
  __aicore__ inline T AllReduceDispatch(T input, Op /*op*/, TagHolder<Tag> h) {
    if (kWarps == 1) { return ::aclcub::WarpReduce(input, Tag{}); }
    T warp_agg = ::aclcub::WarpReduce(input, Tag{});
    if (lane_id_ == 0) { temp_storage_.warp_aggregates[0][warp_id_] = warp_agg; }
    asc_syncthreads();
    return FoldAuto(0, h);  // valid in EVERY thread
  }
  // void tag: arbitrary functor -> generic tree (log2 barriers, all-threads
  // result). Single-barrier / double-buffer promises do NOT apply here.
  template <typename Op>
  __aicore__ inline T AllReduceDispatch(T input, Op op, TagHolder<void>) {
    return ReduceDispatch(input, op, TagHolder<void>{});
  }

  // ----- AllReduce double-buffered. FOLD: 0=auto 1=serial 2=warp -------------
  template <int FOLD, typename Op, typename Tag>
  __aicore__ inline T AllReduceDB(T input, Op /*op*/, TagHolder<Tag> h, int iter) {
    if (kWarps == 1) { return ::aclcub::WarpReduce(input, Tag{}); }
    const int p = iter & 1;   // parity must strictly alternate per reuse (iter CONTRACT)
    T warp_agg = ::aclcub::WarpReduce(input, Tag{});
    if (lane_id_ == 0) { temp_storage_.warp_aggregates[p][warp_id_] = warp_agg; }
    asc_syncthreads();
    if (FOLD == 1) return FoldSerial(p, h);
    if (FOLD == 2) return FoldWarp(p, h);
    return FoldAuto(p, h);
  }
  // void tag: generic tree; iter ignored (own per-call scratch, reuse-safe).
  template <int FOLD, typename Op>
  __aicore__ inline T AllReduceDB(T input, Op op, TagHolder<void>, int /*iter*/) {
    return ReduceDispatch(input, op, TagHolder<void>{});
  }

  // ----- Generic path: arbitrary callable Op (log2 barriers) ------------------
  // Uses its OWN per-call UB scratch (not the caller's TempStorage), so reuse is
  // safe without an extra barrier, but this path is single-buffered and pays
  // log2(BLOCK) barriers — the iter/double-buffer optimization does not apply.
  template <typename Op>
  __aicore__ inline T ReduceDispatch(T input, Op op, TagHolder<void>) {
    __ubuf__ T generic_buf[kBlockThreads];
    generic_buf[linear_tid_] = input;
    asc_syncthreads();
#pragma unroll
    for (int stride = kBlockThreads / 2; stride > 0; stride >>= 1) {
      if (linear_tid_ < stride) {
        generic_buf[linear_tid_] =
            op(generic_buf[linear_tid_], generic_buf[linear_tid_ + stride]);
      }
      asc_syncthreads();
    }
    return generic_buf[0];  // valid in every thread (broadcast via UB)
  }

  _TempStorage& temp_storage_;
  int linear_tid_;
  int warp_id_;
  int lane_id_;
};

// ----------------------------------------------------------------------------
// BlockReduce specialization — ALGORITHM = ACLCUB_BLOCK_REDUCE_ASC_SHFL
// (diagnostic; supports ANY functor; single-buffer)
// ----------------------------------------------------------------------------
template <typename T, int BLOCK_DIM_X>
class BlockReduce<T, BLOCK_DIM_X, ACLCUB_BLOCK_REDUCE_ASC_SHFL> {
 public:
  static constexpr int kBlockThreads = BLOCK_DIM_X;
  static constexpr int kWarpThreads  = kWarpSize;
  static constexpr int kWarps        = (kBlockThreads + kWarpThreads - 1) / kWarpThreads;
  static_assert(kWarps <= kWarpSize,
                "aclcub::BlockReduce ASC_SHFL: kWarps must fit in a single warp.");

  struct _TempStorage { T warp_aggregates[kWarps]; };
  struct TempStorage { _TempStorage data; };

  __aicore__ inline explicit BlockReduce(TempStorage& temp_storage)
      : temp_storage_(temp_storage.data),
        linear_tid_(static_cast<int>(threadIdx.x)),
        warp_id_(linear_tid_ / kWarpThreads),
        lane_id_(linear_tid_ % kWarpThreads) {}

  template <typename Op>
  __aicore__ inline T Reduce(T input, Op op) {
    T val = input;
#pragma unroll
    for (int mask = kWarpThreads / 2; mask > 0; mask /= 2) {
      val = op(val, ::aclcub::AscShflXor(val, mask, kWarpThreads));
    }
    if (kWarps == 1) return val;
    if (lane_id_ == 0) { temp_storage_.warp_aggregates[warp_id_] = val; }
    asc_syncthreads();
    if (linear_tid_ == 0) {
      T agg = temp_storage_.warp_aggregates[0];
#pragma unroll
      for (int i = 1; i < kWarps; ++i) {
        agg = op(agg, temp_storage_.warp_aggregates[i]);
      }
      val = agg;
    }
    return val;
  }

  __aicore__ inline T Sum(T input) { return Reduce(input, ::aclcub::Sum{}); }

 private:
  _TempStorage& temp_storage_;
  int linear_tid_;
  int warp_id_;
  int lane_id_;
};

}  // namespace aclcub

// ============================================================================
// Migration recognizer map list
// ============================================================================
#ifndef ACLCUB_NO_THIRDPARTY_RECOGNIZERS

namespace oneflow { namespace cuda { namespace softmax {
template <typename T> struct SumOp;
template <typename T> struct MaxOp;
} } }  // namespace oneflow::cuda::softmax
namespace aclcub {
template <typename T>
struct OpTagOf< ::oneflow::cuda::softmax::SumOp<T> > { using type = Sum; };
template <typename T>
struct OpTagOf< ::oneflow::cuda::softmax::MaxOp<T> > { using type = Max; };
}  // namespace aclcub

#endif  // ACLCUB_NO_THIRDPARTY_RECOGNIZERS

#endif  // ACL_CUB_ACLCUB_HPP_
