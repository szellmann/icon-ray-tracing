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

#include <owl/owl_host.h>
#include "Params.h"

namespace icon_rt {

/* mapping from our launch params struct to owl's var decl
  that owl can build its shader binding table from: */
OWLVarDecl launchParams_owl[]
= {
   // volume
   { "volume.handle", OWL_GROUP, OWL_OFFSETOF(LaunchParams,volume.handle) },
   { "volume.mode", OWL_INT, OWL_OFFSETOF(LaunchParams,volume.mode) },
   { "volume.cubql.handle", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,volume.cubql.handle) },
   { "volume.cubql.vertices", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,volume.cubql.vertices) },
   { "volume.cubql.indices", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,volume.cubql.indices) },
   { "volume.cubql.perVertex", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,volume.cubql.perVertex) },
   { "volume.cells", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,volume.cells) },
   { "volume.numCells", OWL_INT, OWL_OFFSETOF(LaunchParams,volume.numCells) },
   { "volume.bounds", OWL_USER_TYPE(box3f), OWL_OFFSETOF(LaunchParams,volume.bounds) },
   // volume accel
   { "volume.accel.innerRadius", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,volume.accel.innerRadius) },
   { "volume.accel.outerRadius", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,volume.accel.outerRadius) },
   // xf data
   { "transfunc.valueRange", OWL_USER_TYPE(box1f), OWL_OFFSETOF(LaunchParams,transfunc.valueRange) },
   { "transfunc.values", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,transfunc.values) },
   { "transfunc.size", OWL_INT, OWL_OFFSETOF(LaunchParams,transfunc.size) },
   // camera settings
   { "camera.org", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.org) },
   { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_00) },
   { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_du) },
   { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_dv) },
   // framebuffer
   { "fbPointer",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,fbPointer) },
   { "fbDepth",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,fbDepth) },
   { "accumBuffer",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,accumBuffer) },
   { "accumID",   OWL_INT, OWL_OFFSETOF(LaunchParams,accumID) },
   // lighting
   { "ambientColor", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,ambientColor) },
   { "ambientRadiance", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,ambientRadiance) },
   // render settings
   { "unitDistance", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,unitDistance) },
   { nullptr /* sentinel to mark end of list */ }
};

} // namespace icon_rt


