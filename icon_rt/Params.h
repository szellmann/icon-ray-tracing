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

#pragma once

// common
#include <vecmath.h>
// ours
#include "ICONGrid.h"

using namespace vecmath;

// ========================================================
// structs with trivial layout, no default init, etc.
// to safely cross host/device borders
// ========================================================
namespace icon_rt {

struct Volume {
#ifdef RTCORE
  OptixTraversableHandle handle;
  bool useTriangles;
#endif
  ICONCell *cells;
  int numCells;
  box3f bounds;

  // the most simple accelerator imaginable
  // for this kind of data, just two spheres
  // forming a shell around the icon elements;
  // everything inside the inner and outside
  // the outer sphere is considered empty
  struct {
    float innerRadius, outerRadius;
  } accel;
};

struct Transfunc {
  box1f  valueRange;
  vec4f *values;
  int size;
};

struct ICONTriangleGeom
{
  /*! array/buffer of vertex indices */
  vec3i *index;
  /*! array/buffer of vertex positions */
  vec3f *vertex;
};

struct LaunchParams {
  // volume:
  Volume volume;

  // transfunc:
  Transfunc transfunc;

  // camera:
  struct {
    vec3f org;
    vec3f dir_00;
    vec3f dir_du;
    vec3f dir_dv;
  } camera;

  // framebuffer:
  uint32_t *fbPointer;
  float    *fbDepth;
  vec4f    *accumBuffer;
  int       accumID;

  // lighting:
  vec3f ambientColor;
  float ambientRadiance;

  // DVR:
  float unitDistance;
};

} // namespace icon_rt


