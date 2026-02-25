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

// std
#include <vector>
// ours
#include <vecmath.h>

namespace dvr_course {

// import into "dvr_course":
using namespace vecmath;

struct Transfunc
{
  float opacity{1.f};
  box1f valueRange{0.f,1.f};
  box1f relRange{0.f,1.f};
  // data:
  vec4f *rgbaLUT{nullptr};
  int size{0};

  Transfunc() = default;
  Transfunc(const Transfunc &other);
  Transfunc(Transfunc &&other);
  ~Transfunc();
  Transfunc &operator=(const Transfunc &other);
  Transfunc &operator=(Transfunc &&other);

  // set LUT (device upload)
  void setLUT(const std::vector<vec4f> &lut);
  // get LUT (download from device)
  std::vector<vec4f> getLUT() const;
};

} // namespace dvr_course
