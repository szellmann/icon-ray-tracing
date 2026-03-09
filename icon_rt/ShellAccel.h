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
  vec3i dims;
  box3f sphericalBounds;
  box1f *valueRanges;
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

#define g_latBounds box1f{-M_PI/2.f,M_PI/2.f}
#define g_lonBounds box1f{-M_PI,M_PI}

inline __device__
float normalizeLat(float lat) {
  while (lat<g_latBounds.lower) lat += g_latBounds.size();
  while (lat>g_latBounds.upper) lat += g_latBounds.size();
  return lat;
}

inline __device__
float normalizeLon(float lon) {
  while (lon<g_lonBounds.lower) lon += g_lonBounds.size();
  while (lon>g_lonBounds.upper) lon += g_lonBounds.size();
  return lon;
}

// this returns unbounded coordinates (can be negative or great than dims)
// Useful for DDA, but no so for array accesses
inline __device__
vec3i projectToSphericalGrid(const vec3f sph, const vec3i dims) {
  return{0, // todo
      (sph.y-g_latBounds.lower)/g_latBounds.size()*(dims.y-1),
      (sph.z-g_lonBounds.lower)/g_lonBounds.size()*(dims.z-1)};
}

// normalize unbounded coordinates to [0:dims)
inline __device__
vec3i normalizeGridCoord(vec3i coord, const vec3i dims) {
  while (coord.x<0) coord.x += dims.x;
  while (coord.x>=dims.x) coord.x -= dims.x;
  while (coord.y<0) coord.y += dims.y;
  while (coord.y>=dims.y) coord.y -= dims.y;
  while (coord.z<0) coord.z += dims.z;
  while (coord.z>=dims.z) coord.z -= dims.z;
  return coord;
}

template<typename Func>
inline __device__
void sdda(Ray ray, const ShellAccel &accel, const Func &func, bool dbg=false) {
  float t1,t2,t3,t4;
  bool s1 = intersectSphere(ray,accel.sphericalBounds.upper.x,t1,t4);
  bool s2 = intersectSphere(ray,accel.sphericalBounds.lower.x,t2,t3);

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
    vec3f P1 = ray.org+ray.dir*ranges[i].lower;
    vec3f P2 = ray.org+ray.dir*ranges[i].upper;
    vec3f SP1 = toSpherical(P1);
    vec3f SP2 = toSpherical(P2);

    const float latInc = (M_PI)/float(accel.dims.y);
    const float lonInc = (M_PI*2)/float(accel.dims.z);

    vec3i cellID
        = projectToSphericalGrid(toSpherical(ray.eval(ranges[i].lower)),accel.dims);

    // Cell increment
    const vec3i step = {
      SP1.x < SP2.x ? 1 : -1, // rad
      SP1.y < SP2.y ? 1 : -1, // lat
      SP1.z < SP2.z ? 1 : -1 // lon
    };

    // Stop when we step beyond the outermost cell
    const vec3i stop
        = projectToSphericalGrid(toSpherical(ray.eval(ranges[i].upper)),accel.dims)+step;

    // Increment in world space
    float latOff = (cellID.y+step.y)*latInc;
    float lonOff = (cellID.z+step.z)*lonInc;
    Plane latPlane = makePlane(vec3f(0.f),
        toCartesian(vec3f(0.f,latOff,g_latBounds.lower)),
        toCartesian(vec3f(0.f,latOff,g_latBounds.upper)));
    Plane lonPlane = makePlane(vec3f(0.f),
        toCartesian(vec3f(0.f,g_lonBounds.lower,lonOff)),
        toCartesian(vec3f(0.f,g_lonBounds.upper,lonOff)));
    vec3f tnext = {
      ranges[i].upper, // todo
      evalPlane(latPlane,ray.eval(ranges[i].lower)),
      evalPlane(lonPlane,ray.eval(ranges[i].lower))
    };

    float t = ranges[i].lower;
    while (1) {
      vec3f P = ray.org+ray.dir*t;

      float t1 = FLT_MAX;
      if (tnext.x<t1 && tnext.x>=t) t1 = tnext.x;
      if (tnext.y<t1 && tnext.y>=t) t1 = tnext.y;
      if (tnext.z<t1 && tnext.z>=t) t1 = tnext.z;

      int leafID = linearIndex(normalizeGridCoord(cellID,accel.dims),accel.dims);
      if (!func(leafID,t,t1)) return;

      const float t_closest = reduce_min(tnext);
      if (tnext.x == t_closest) {
        cellID.x += step.x;
        if (cellID.x==stop.x) {
          break;
        }
      }
      if (tnext.y == t_closest) {
        cellID.y += step.y;
        if (cellID.y==stop.y) {
          break;
        }
        Plane plane = makePlane(vec3f(0.f),
            toCartesian(vec3f(0.f,cellID.y*latInc,g_latBounds.lower)),
            toCartesian(vec3f(0.f,cellID.y*latInc,g_latBounds.upper)));
        tnext.y = evalPlane(plane,P);
      }
      if (tnext.z == t_closest) {
        cellID.z += step.z;
        if (cellID.z==stop.z) {
          break;
        }
        Plane plane = makePlane(vec3f(0.f),
            toCartesian(vec3f(0.f,g_lonBounds.lower,cellID.z*lonInc)),
            toCartesian(vec3f(0.f,g_lonBounds.upper,cellID.z*lonInc)));
        tnext.z = evalPlane(plane,P);
      }
      t = t_closest;
    }
    // if (dbg) {
    //   auto stop2 = projectToSphericalGrid(toSpherical(ray.eval(t)),accel.dims);
    //   printf("%i,%i,%i -- %i,%i,%i\n",
    //     stop.x,
    //     stop.y,
    //     stop.z,
    //     stop2.x,
    //     stop2.y,
    //     stop2.z);
    // }
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


