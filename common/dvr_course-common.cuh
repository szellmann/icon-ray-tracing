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

// ========================================================
// Common header for use in *device* code
// ========================================================

#pragma once

#ifndef RTCORE
#include <iostream>
#else
#include <owl/owl_device.h>
#endif
#include <cstdint>
#include "dvr_course-common-both.h"
#include "vecmath.h"

namespace dvr_course {
using namespace vecmath;
} // dvr_course

#ifndef __CUDACC__
#define __constant__
#define __shared__
#endif

namespace dvr_course {
#ifdef RTCORE
inline __device__ const vec2i getLaunchIndex(void) {
  auto li = owl::getLaunchIndex();
  return {li.x,li.y};
}

inline __device__ const vec2i getLaunchDims(void) {
  auto ld = owl::getLaunchDims();
  return {ld.x,ld.y};
}

inline __device__ bool debug(void) {
  const auto launchIndex = getLaunchIndex();
  const auto launchDims = getLaunchDims();
  return launchIndex.x == launchDims.x/2 && launchIndex.y == launchDims.y/2;
}
#define RAYGEN_PROGRAM OPTIX_RAYGEN_PROGRAM
#else
const vec2i getLaunchIndex(void);
const vec2i getLaunchDims(void);
const bool debug(void);
#define RAYGEN_PROGRAM(name) void name
#endif

} // dvr_course


