/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "CUDA2DPP.h"

using SEC = runtime::CUDA_RUNTIME_API_SECTIONS;

// Map of all CUDA Runtime API functions
const std::map<llvm::StringRef, dppCounter> CUDA_RUNTIME_FUNCTION_MAP = [] {
  std::map<llvm::StringRef,  dppCounter> m;
  m["cudaMalloc"]                                              = {"aclrtMalloc",                                            CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaFree"]                                              = {"aclrtFree",                                               CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpy"]                                              = {"aclrtMemcpy",                                              CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["free"] = {"aclrtFreeHost", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // TODO: Add cuda runtime API functions here  
  return m;
}();
