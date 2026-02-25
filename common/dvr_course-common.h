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
// Common header for use in *host* code
// ========================================================

#pragma once

// std
#include <iostream>
#include <string>
#include <vector>
// ours
#include "dvr_course-common-both.h"
#include "buffer.h"
#include "fb.h"
#include "vecmath.h"

namespace dvr_course {
using namespace vecmath;


inline bool endsWith(const std::string &s, const std::string &suffix) {
  if (s.length() < suffix.length())
    return false;

  return s.substr(s.size()-suffix.size(),suffix.size()) == suffix;
}

inline void resampleLUT(std::vector<vec4f> &dst, const std::vector<vec4f> &src) {
  int srcDims = (int)src.size();
  int dstDims = (int)dst.size();

  // The user-provided colors
  const vec4f *colors = src.data();

  // Updated colors
  vec4f *updated = dst.data();

  // Lerp colors and alpha
  for (int i = 0; i < dstDims; ++i) {
    float indexf = i / (float)(dstDims) * (srcDims-1);
    int indexa = (int)indexf;
    int indexb = std::min(indexa+1, srcDims-1);
    vec3f rgb1(colors[indexa].x, colors[indexa].y, colors[indexa].z);
    float alpha1 = colors[indexa].w;
    vec3f rgb2(colors[indexb].x, colors[indexb].y, colors[indexb].z);
    float alpha2 = colors[indexb].w;
    float frac = indexf-indexa;

    vec3f rgb = lerp(rgb1, rgb2, 1.f-frac);
    float alpha = lerp(alpha1, alpha2, 1.f-frac);

    updated[i] = vec4f(rgb,alpha);
  }
}

} // dvr_course

#include "camera.h"
#include "pipeline.h"

