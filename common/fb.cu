// ======================================================================== //
// Copyright 2025-2025 Stefan Zellmann                                      //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "dvr_course-common.cuh"
#include "fb.h"

namespace dvr_course {

Frame::Frame(int w, int h) : width(w), height(h)
{
#ifdef RTCORE
  cudaMalloc(&fbPointer,w*h*sizeof(uint32_t));
  cudaMalloc(&fbDepth,w*h*sizeof(float));
  cudaMalloc(&accumBuffer,w*h*sizeof(vec4f));
#else
  fbPointer   = (uint32_t *)std::malloc(w*h*sizeof(uint32_t));
  fbDepth     = (float *)std::malloc(w*h*sizeof(float));
  accumBuffer = (vec4f *)std::malloc(w*h*sizeof(vec4f));
#endif
}

Frame::~Frame()
{
#ifdef RTCORE
  cudaFree(fbPointer);
  cudaFree(fbDepth);
  cudaFree(accumBuffer);
#else
  std::free(fbPointer);
  std::free(fbDepth);
  std::free(accumBuffer);
#endif
}

void Frame::resize(int w, int h)
{
  width = w; height = h;
#ifdef RTCORE
  //cudaFree(fbPointer);
  //cudaFree(fbDepth);
  //cudaFree(accumBuffer);
  cudaMalloc(&fbPointer,w*h*sizeof(uint32_t));
  cudaMalloc(&fbDepth,w*h*sizeof(float));
  cudaMalloc(&accumBuffer,w*h*sizeof(vec4f));
#else
  std::free(fbPointer);
  std::free(fbDepth);
  std::free(accumBuffer);
  fbPointer   = (uint32_t *)std::malloc(w*h*sizeof(uint32_t));
  fbDepth     = (float *)std::malloc(w*h*sizeof(float));
  accumBuffer = (vec4f *)std::malloc(w*h*sizeof(vec4f));
#endif

}

} // namespace dvr_course
