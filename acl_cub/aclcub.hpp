/*
 *    #include <acl_cub/aclcub.hpp>
 * Provides:
 *   aclcub::Sum / aclcub::Max / aclcub::Min                (op tags)
 *   aclcub::BlockReduce<T, BLOCK_DIM_X>::TempStorage       (shared-mem alias)
 *   aclcub::BlockReduce<T, BLOCK_DIM_X>::Reduce(val, op)
 *   aclcub::BlockReduce<T, BLOCK_DIM_X>::Sum(val)
 *
 * Two execution paths, dispatched at compile time so unused paths are not
 * instantiated:
 *   * Fast path: when Op publicly derives from aclcub::Sum/Max/Min OR is
 *       registered in the recognizer map list at the bottom of this file:
 *         (1) intra-warp AllReduce via asc_reduce_add / asc_reduce_max /
 *             asc_reduce_min
 *         (2) lane 0 of each warp publishes its warp aggregate to UB
 *         (3) asc_syncthreads()
 *         (4) warp 0 folds the per-warp aggregates with another
 *             asc_reduce_* (one barrier, no serial fan-in)
 *   * Generic path (any other callable): SMEM tree reduction calling
 *       op(a, b). Preserves cub's "any binary functor works" contract at
 *       the cost of log2(BLOCK_DIM_X) extra barriers.
 */

#ifndef ACL_CUB_ACLCUB_HPP_
#define ACL_CUB_ACLCUB_HPP_

#include <cstdint>
#include <simt_api/device_warp_functions.h>
#include <simt_api/device_sync_functions.h>

namespace aclcub {

#ifndef ACLCUB_WARP_SIZE
#define ACLCUB_WARP_SIZE 32
#endif
constexpr int kWarpSize = ACLCUB_WARP_SIZE;

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

template <bool C, typename T, typename F> struct Cond { using type = T; };
template <typename T, typename F> struct Cond<false, T, F> { using type = F; };

template <typename Op>
struct OpTagOf {
  using type =
      typename Cond<__is_base_of(Sum, Op), Sum,
      typename Cond<__is_base_of(Max, Op), Max,
      typename Cond<__is_base_of(Min, Op), Min,
                    void>::type>::type>::type;
};

template <typename Tag> struct TagHolder {};

// BlockReduce<T, BLOCK_DIM_X>
template <typename T, int BLOCK_DIM_X>
class BlockReduce {
 public:
  static constexpr int kBlockThreads = BLOCK_DIM_X;
  static constexpr int kWarpThreads  = kWarpSize;
  static constexpr int kWarps        = (kBlockThreads + kWarpThreads - 1) / kWarpThreads;
  static_assert(kWarps <= kWarpSize,
                "aclcub::BlockReduce: kWarps must fit in a single warp for the "
                "second-level reduction. Cap BLOCK_DIM_X at kWarpSize*kWarpSize.");

  // One slot per warp, used by the asc_reduce_* fast path. Lives in UB.
  struct _TempStorage { T warp_aggregates[kWarps]; };
  struct TempStorage { _TempStorage data; };

  __aicore__ inline explicit BlockReduce(TempStorage& temp_storage)
      : temp_storage_(temp_storage.data),
        linear_tid_(static_cast<int>(threadIdx.x)),
        warp_id_(linear_tid_ / kWarpThreads),
        lane_id_(linear_tid_ % kWarpThreads) {}

  template <typename Op>
  __aicore__ inline T Reduce(T input, Op op) {
    using OpTag = typename OpTagOf<Op>::type;
    return ReduceDispatch(input, op, TagHolder<OpTag>{});
  }

  __aicore__ inline T Sum(T input) {
    return ReduceDispatch(input, ::aclcub::Sum{}, TagHolder<::aclcub::Sum>{});
  }

 private:
  // ----- Fast path: Op routed to one of Sum/Max/Min -------------------------
  template <typename Op, typename Tag>
  __aicore__ inline T ReduceDispatch(T input, Op /*op*/, TagHolder<Tag>) {
    // Single-warp block: the warp AllReduce already spans the whole block, so
    // skip shared memory and the barrier entirely (kWarps is constexpr, so for
    // kWarps > 1 this branch is elided). Best config for a warp-sized block.
    if (kWarps == 1) { return ::aclcub::WarpReduce(input, Tag{}); }

    // (1) Intra-warp AllReduce via the Ascend primitive.
    T warp_agg = ::aclcub::WarpReduce(input, Tag{});

    // (2) Lane 0 of each warp publishes its warp aggregate.
    if (lane_id_ == 0) { temp_storage_.warp_aggregates[warp_id_] = warp_agg; }
    asc_syncthreads();

    // (3) Warp 0 folds the WARPS aggregates with the same primitive.
    if (warp_id_ == 0) {
      T v = (lane_id_ < kWarps)
                ? temp_storage_.warp_aggregates[lane_id_]
                : Identity<T, Tag>::value();
      warp_agg = ::aclcub::WarpReduce(v, Tag{});
    }
    return warp_agg;  // valid in thread 0 (and across warp 0)
  }

  // ----- Generic path: arbitrary callable Op --------------------------------
  // log2(BLOCK_DIM_X) shared-memory tree reduction. Used when Op cannot be
  // classified into {Sum,Max,Min}. Preserves cub's semantics at the cost of
  // higher latency and BLOCK_DIM_X * sizeof(T) of static UB.
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

}  // namespace aclcub

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

