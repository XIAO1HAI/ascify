#!/bin/bash
# Build + run the double-buffered AllReduce head-to-head (V0/V1/V2) + correctness.
# Candidate header lives in ./acl_cub/aclcub.hpp (this dir), included BEFORE the
# Repo copy via -I. Pins an idle card (default 5; cards 1/4/5 idle on 2026-06-12).
set -u
WORKDIR=/home/Yhaibiao/CUDA_to_Ascend/aclcub/0612_aclcub
cd "$WORKDIR"
source /home/Yhaibiao/cann_path_cloud_0325/cann-9.0.0/set_env.sh
export ASCEND_CUSTOM_PATH=/home/Yhaibiao/cann_path_cloud_0325/cann-9.0.0/
export CANN_PATH=/home/Yhaibiao/cann_path_cloud_0325/cann-9.0.0/

echo "[build] ccec compiling bench_allreduce_db.cce ..."
${CANN_PATH}/tools/bisheng_compiler/bin/ccec \
  -x dpp --cce-aicore-arch=dav-c310-vec -std=c++11 -D__NPU_ARCH__=3510 \
  bench_allreduce_db.cce \
  -I "$WORKDIR" \
  -I${CANN_PATH}/include \
  -I${CANN_PATH}/include/ascendc/host_api \
  -I${CANN_PATH}/compiler/ascendc/include/highlevel_api \
  -I${CANN_PATH}/compiler/tikcpp/tikcfw \
  -I${CANN_PATH}/compiler/tikcpp/tikcfw/lib \
  -I${CANN_PATH}/compiler/tikcpp/tikcfw/lib/matmul \
  -I${CANN_PATH}/compiler/tikcpp/tikcfw/impl \
  -I${CANN_PATH}/compiler/tikcpp/tikcfw/interface \
  -I /home/Yhaibiao/CUDA_to_Ascend/Repo/oneflow/ \
  -I ${CANN_PATH}/x86_64-linux/asc/include/ \
  -L${CANN_PATH}/lib64 \
  -lascendc_runtime -lascendcl -lruntime -lregister -lerror_manager \
  -lprofapi -lascendalog -lmmpa -lascend_dump -lc_sec \
  -g -gdwarf-4 -lstdc++ \
  -O3 \
  -mllvm -enable-load-pre=true \
  -mllvm -vectorize-slp=true \
  -o bench_allreduce_db
build_rc=$?
echo "[build] rc=${build_rc}"

export LD_LIBRARY_PATH=${CANN_PATH}/lib64:${CANN_PATH}/runtime/lib64:${LD_LIBRARY_PATH}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-5}
echo "[run] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
if [ "${build_rc}" -eq 0 ] && [ -x ./bench_allreduce_db ]; then
  ./bench_allreduce_db
  echo "[run] exit=$?"
else
  echo "[run] skipped (compile failed rc=${build_rc})"
  exit 1
fi
