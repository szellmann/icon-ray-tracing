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

// std
#include <cstring> // memcpy
// ours
#include "transfunc.h"

namespace dvr_course {

Transfunc::Transfunc(const Transfunc &other)
  : opacity(other.opacity),
    valueRange(other.valueRange),
    relRange(other.relRange),
    size(other.size) {
  if (&other != this) {
#ifdef RTCORE
    cudaMalloc(&rgbaLUT,sizeof(rgbaLUT[0])*size);
    cudaMemcpy(rgbaLUT,other.rgbaLUT,sizeof(rgbaLUT[0])*size,cudaMemcpyDefault);
#else
    rgbaLUT = (vec4f *)std::malloc(sizeof(rgbaLUT[0])*size);
    std::memcpy(rgbaLUT,other.rgbaLUT,sizeof(rgbaLUT[0])*size);
#endif
  }
}

Transfunc::Transfunc(Transfunc &&other)
  : opacity(other.opacity),
    valueRange(other.valueRange),
    relRange(other.relRange),
    size(other.size) {
  if (&other != this) {
#ifdef RTCORE
    cudaMalloc(&rgbaLUT,sizeof(rgbaLUT[0])*size);
    cudaMemcpy(rgbaLUT,other.rgbaLUT,sizeof(rgbaLUT[0])*size,cudaMemcpyDefault);
#else
    rgbaLUT = (vec4f *)std::malloc(sizeof(rgbaLUT[0])*size);
    std::memcpy(rgbaLUT,other.rgbaLUT,sizeof(rgbaLUT[0])*size);
#endif
    other.opacity = 0.f;
    other.valueRange = {0.f,0.f};
    other.rgbaLUT = nullptr;
    other.size = 0;
  }
}

Transfunc &Transfunc::operator=(const Transfunc &other) {
  if (&other != this) {
    opacity = other.opacity;
    valueRange = other.valueRange;
    relRange = other.relRange;
    size = other.size;
#ifdef RTCORE
    cudaMalloc(&rgbaLUT,sizeof(rgbaLUT[0])*size);
    cudaMemcpy(rgbaLUT,other.rgbaLUT,sizeof(rgbaLUT[0])*size,cudaMemcpyDefault);
#else
    rgbaLUT = (vec4f *)std::malloc(sizeof(rgbaLUT[0])*size);
    std::memcpy(rgbaLUT,other.rgbaLUT,sizeof(rgbaLUT[0])*size);
#endif
  }
  return *this;
}

Transfunc &Transfunc::operator=(Transfunc &&other) {
  if (&other != this) {
    opacity = other.opacity;
    valueRange = other.valueRange;
    relRange = other.relRange;
    size = other.size;
#ifdef RTCORE
    cudaMalloc(&rgbaLUT,sizeof(rgbaLUT[0])*size);
    cudaMemcpy(rgbaLUT,other.rgbaLUT,sizeof(rgbaLUT[0])*size,cudaMemcpyDefault);
#else
    rgbaLUT = (vec4f *)std::malloc(sizeof(rgbaLUT[0])*size);
    std::memcpy(rgbaLUT,other.rgbaLUT,sizeof(rgbaLUT[0])*size);
#endif
    other.opacity = 0.f;
    other.valueRange = {0.f,0.f};
    other.rgbaLUT = nullptr;
    other.size = 0;
  }
  return *this;
}

Transfunc::~Transfunc() {
#ifdef RTCORE
  cudaFree(rgbaLUT);
#else
  std::free(rgbaLUT);
#endif
}

void Transfunc::setLUT(const std::vector<vec4f> &lut) {
  if (lut.size() != size) {
#ifdef RTCORE
    cudaFree(rgbaLUT);
    cudaMalloc(&rgbaLUT,sizeof(lut[0])*lut.size());
#else
    std::free(rgbaLUT);
    rgbaLUT = (vec4f *)std::malloc(sizeof(lut[0])*lut.size());
#endif
    size = (int)lut.size();
  }

#ifdef RTCORE
  cudaMemcpy(rgbaLUT,lut.data(),sizeof(lut[0])*lut.size(),
             cudaMemcpyHostToDevice);
#else
  memcpy(rgbaLUT,lut.data(),sizeof(lut[0])*lut.size());
#endif
}

std::vector<vec4f> Transfunc::getLUT() const {
  if (size <= 0) return {};

  std::vector<vec4f> lut(size);
#ifdef RTCORE
  cudaMemcpy(lut.data(),rgbaLUT,sizeof(lut[0])*size,
             cudaMemcpyDeviceToHost);
#else
  std::memcpy(lut.data(),rgbaLUT,sizeof(lut[0])*size);
#endif
  return lut;
}

} // namespace dvr_course
