export LLVM_PROJECT_PATH="/llvm-project"
export ASCIFY_PATH="/ascify/build"
export CUDA_PATH="/usr/local/cuda"
$ASCIFY_PATH/ascify-clang $1 --clang-resource-directory=$LLVM_PROJECT_PATH/build/lib/clang/23 --cuda-path=$CUDA_PATH