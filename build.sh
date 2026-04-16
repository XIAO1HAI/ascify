# Compiler for *this* build: must target the machine you run CMake on.
# A stage-1 LLVM that only enables AArch64/ARM/NVPTX (no X86) and leaves
# LLVM_DEFAULT_TARGET_TRIPLE empty cannot compile native x86_64 objects; use
# system clang here, or rebuild LLVM with X86 (or Native) and a proper
# LLVM_DEFAULT_TARGET_TRIPLE, then point ASCIFY_CC at that clang.
: "${ASCIFY_CC:=$LLVM_PROJECT_PATH/build/bin/clang}"
: "${ASCIFY_CXX:=$LLVM_PROJECT_PATH/build/bin/clang++}"

cmake -G Ninja   -DCMAKE_INSTALL_PREFIX=$(pwd)/../ascify_install   -DCMAKE_BUILD_TYPE=Release   \
    -DCMAKE_PREFIX_PATH=$LLVM_PROJECT_PATH/build   \
    -DCMAKE_C_COMPILER="$ASCIFY_CC" \
    -DCMAKE_CXX_COMPILER="$ASCIFY_CXX" \
    -DCMAKE_LINKER=$LLVM_PROJECT_PATH/build/bin/lld \
    ../
