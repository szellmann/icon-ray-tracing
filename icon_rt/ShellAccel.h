// ======================================================================== //
// Copyright 2025-2026 Stefan Zellmann                                      //
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

#include "ICONGrid.h" // conversion functions from spherical to Cartesian, etc.
namespace icon_rt {

struct ShellAccel {
  box1f *radialBounds, *latBounds, *lonBounds;
  float *maxOpacities;
};


// ========================================================
// Sphere intersection, origin at (0,0,0)
// ========================================================

inline __device__
bool intersectSphere(const Ray &ray, float radius, float &tnear, float &tfar) {
  float A = dot(ray.dir,ray.dir);
  float B = dot(ray.dir,ray.org) * 2.f;
  float C = dot(ray.org,ray.org) - radius*radius;

  float d = B*B - 4.f*A*C;
  if (d < 0.f) return false;

  d = sqrtf(d);

  float q = B < 0.f ? -0.5f * (B-d) : -0.5f * (B+d);

  float t1 = q/A;
  float t2 = C/q;

  tnear = fminf(t1,t2);
  tfar  = fmaxf(t1,t2);
  return true;
}

template<typename Func>
inline __device__
void traverseShellAccel(Ray ray, const ShellAccel &accel, const Func &func) {
  float t1,t2,t3,t4;
  bool s1 = intersectSphere(ray,accel.radialBounds->upper,t1,t4);
  bool s2 = intersectSphere(ray,accel.radialBounds->lower,t2,t3);

  if (!s1 && !s2) return;
  if (t4 < ray.tmin) return;

  box1f ranges[2] = {
    {INFINITY,-INFINITY},
    {INFINITY,-INFINITY},
  };

  // outer sphere hit, but inner was missed:
  if (s1 && !s2) {
    ranges[0] = {t1,t4};
  }
  // inside front segment:
  else if (ray.tmin < t2) {
    ranges[0] = {t1,t2};
    ranges[1] = {t3,t4};
  }
  // inside back segment:
  else { 
    ranges[0] = {t3,t4};
  }

  for (int i=0; i<2; ++i) {
    if (ranges[i].empty()) break;
    int leafID=0;
    if (!func(leafID,ranges[i].lower,ranges[i].upper)) break;
  }

  // vec3f P1 = toSpherical(ray.org+ray.dir*t1);
  // vec3f P2 = toSpherical(ray.org+ray.dir*t2);
  // vec3f P3 = toSpherical(ray.org+ray.dir*t3);
  // vec3f P4 = toSpherical(ray.org+ray.dir*t4);

  // box1f latBounds = *accel.latBounds;
  // if (!(latBounds.contains(P1.y)||latBounds.contains(P2.y)||
  //       latBounds.contains(P3.y)||latBounds.contains(P4.y)))
  //   return false;

  // box1f lonBounds = *accel.lonBounds;
  // if (!(lonBounds.contains(P1.z)||lonBounds.contains(P2.z)||
  //       lonBounds.contains(P3.z)||lonBounds.contains(P4.z)))
  //   return false;
}

}


