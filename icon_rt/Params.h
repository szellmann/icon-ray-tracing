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

// cuBQL
#include "cuBQL/traversal/fixedBoxQuery.h"
// common
#include <vecmath.h>
// ours
#include "ICONGrid.h"

using namespace vecmath;

#define USER_GEOM_MODE    0
#define TRIANGLE_MODE     1
#define CUBQL_MODE        2

#define SPHERE_ACCEL_MODE 0
#define GRID_ACCEL_MODE   1

// ========================================================
// structs with trivial layout, no default init, etc.
// to safely cross host/device borders
// ========================================================
namespace icon_rt {

using bvh_t  = cuBQL::BinaryBVH<float,3>;

struct Grid {
  box1f *valueRanges;
  vec3i dims;
  box3f worldBounds;
  float *maxOpacities;
};

struct Volume {
#ifdef RTCORE
  OptixTraversableHandle handle;
  struct {
    bvh_t *handle;
    vec3f *vertices;
    int *indices;
    float *perVertex;
  } cubql;
  int mode; // 0=userGeom, 1=triangles, 2=umesh/cuBQL
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
  // grid accel for testing against:
  Grid gridAccel;
  int accelMode;
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


