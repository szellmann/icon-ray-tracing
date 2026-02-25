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
// Common header with helpers (both host _and_ device)
// ========================================================

#pragma once

#include "vecmath.h"

namespace dvr_course {

using namespace vecmath;

inline __device__ float linear_to_srgb(float x) {
  if (x <= 0.0031308f) {
    return 12.92f * x;
  }
  return 1.055f * powf(x, 1.f/2.4f) - 0.055f;
}

// ==================================================================
// RNG
// ==================================================================

template<unsigned int N=4>
struct LCG
{
  inline __host__ __device__ LCG()
  { /* intentionally empty so we can use it in device vars that
       don't allow dynamic initialization (ie, PRD) */
  }
  inline __host__ __device__ LCG(unsigned int val0, unsigned int val1)
  { init(val0,val1); }

  inline __host__ __device__ LCG(const vec2i &seed)
  { init((unsigned)seed.x,(unsigned)seed.y); }
  inline __host__ __device__ LCG(const vec2ui &seed)
  { init(seed.x,seed.y); }
  
  inline __host__ __device__ void init(unsigned int val0, unsigned int val1)
  {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;
  
    for (unsigned int n = 0; n < N; n++) {
      s0 += 0x9e3779b9;
      v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
      v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }
    state = v0;
  }

  // Generate random unsigned int in [0, 2^24)
  inline __host__ __device__ float operator() ()
  {
    const uint32_t LCG_A = 1664525u;
    const uint32_t LCG_C = 1013904223u;
    state = (LCG_A * state + LCG_C);
    return (state & 0x00FFFFFF) / (float) 0x01000000;
  }

  // For compat. with visionaray
  inline __host__ __device__ float next()
  {
    return operator()();
  }

  uint32_t state;
};

typedef LCG<4> Random;
inline __device__ uint32_t make_8bit(const float f)
{
  return fminf(255,fmaxf(0,int(f*256.f)));
}

inline __device__ uint32_t make_rgba(const vecmath::vec3f color)
{
  return
    (make_8bit(color.x) << 0) +
    (make_8bit(color.y) << 8) +
    (make_8bit(color.z) << 16) +
    (0xffU << 24);
}

inline __device__ uint32_t make_rgba(const vecmath::vec4f color)
{
  return
    (make_8bit(color.x) << 0) +
    (make_8bit(color.y) << 8) +
    (make_8bit(color.z) << 16) +
    (make_8bit(color.w) << 24);
}

} // dvr_course


