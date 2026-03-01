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

// common
#include <dvr_course-common.cuh>
// icon_rt
#include "DDA.h"
#include "Params.h"
#include "UElems.h"

using namespace dvr_course;

// ========================================================
// device code for example 05: hey_icon
// ========================================================
namespace icon_rt {

extern "C" __constant__ LaunchParams optixLaunchParams;

// ========================================================
// Helpers
// ========================================================
inline  __device__ Ray generateRay(const vec2f screen, Random &rnd)
{
  auto &lp = optixLaunchParams;
  vec3f org = lp.camera.org;
  vec3f dir
    = lp.camera.dir_00
    + (screen.u+rnd()) * lp.camera.dir_du
    + (screen.v+rnd()) * lp.camera.dir_dv;
  dir = normalize(dir);
  if (fabsf(dir.x) < 1e-5f) dir.x = 1e-5f;
  if (fabsf(dir.y) < 1e-5f) dir.y = 1e-5f;
  if (fabsf(dir.z) < 1e-5f) dir.z = 1e-5f;
  return Ray(org,dir,0.f,1e10f);
}

#ifdef RTCORE
struct PRD {
  float value;
  unsigned primID;
};
#endif

inline __device__ bool sampleVolume(const Volume &vol, vec3f pos, float &value)
{
#ifdef RTCORE
  if (vol.mode==TRIANGLE_MODE) {
    PRD prd;
    prd.value = 0.f;
    prd.primID = ~0u;
    owl::Ray ray;
    ray.origin = owl::vec3f(pos.x,pos.y,pos.z);
    ray.direction = -normalize(ray.origin);
    owl::traceRay(vol.handle,ray,prd,OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES);
    if (prd.primID != ~0u) {
      const ICONCell &cell = vol.cells[prd.primID];
      const vec3f spherical = toSpherical(pos);
      if (spherical.x < cell.height[0] || spherical.x > cell.height[cell.numLayers])
        return false;
      value = cell.getValue(spherical.x);
      return true;
    }
  } else if (vol.mode==USER_GEOM_MODE) {
    PRD prd;
    prd.value = 0.f;
    prd.primID = ~0u;
    owl::Ray ray;
    ray.origin = owl::vec3f(pos.x,pos.y,pos.z);
    ray.direction = owl::vec3f(1.f);
    ray.tmin = ray.tmax = 0.f;
    owl::traceRay(vol.handle,ray,prd,OPTIX_RAY_FLAG_DISABLE_ANYHIT);
    if (prd.primID != ~0u) {
      value = prd.value;
      return true;
    }
  } else if (vol.mode==CUBQL_MODE) {
    typename bvh_t::box_t box; box.lower = box.upper = {pos.x,pos.y,pos.z};
    bool hit{false};
    auto lambda = [&vol,pos,&hit,&value]
      (const uint32_t primID)
    {
      const int *I = vol.cubql.indices + primID*6;
      const float v = vol.cubql.perVertex[primID*6+0];
      // Hack direction vector into w:
      const vec4f v0(vol.cubql.vertices[I[0]],v);
      const vec4f v1(vol.cubql.vertices[I[1]],v);
      const vec4f v2(vol.cubql.vertices[I[2]],v);
      const vec4f v3(vol.cubql.vertices[I[3]],v);
      const vec4f v4(vol.cubql.vertices[I[4]],v);
      const vec4f v5(vol.cubql.vertices[I[5]],v);
      if (intersectWedgeEXT(value,pos,v0,v1,v2,v3,v4,v5)) {
        hit = true;
        value = v;
        return CUBQL_TERMINATE_TRAVERSAL;
      }
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    if (hit) printf("%f\n",value);
    cuBQL::fixedBoxQuery::forEachPrim(lambda,*vol.cubql.handle,box);
    return hit;
  }
#else
  // on non-RT hardware we resort to just linearly
  // iterating over all primitives (veeeryy slow...)
  for (unsigned i=0; i<vol.numCells; ++i) {
    if (sample(vol.cells[i],pos,value))
      return true;
  }
#endif
  return false;
}

inline __device__ vec4f postClassify(Transfunc tf, float v)
{
  v = (v - tf.valueRange.lower) / (tf.valueRange.upper - tf.valueRange.lower);
  int idx = v*(tf.size);
  float frac = (v*tf.size)-idx;
  vec4f v1 = tf.values[clamp(idx,0,tf.size-1)];
  vec4f v2 = tf.values[clamp(idx+1,0,tf.size-1)];
  return v1*frac+v2*(1.f-frac);
}

inline __device__ float woodcockTracking(const Ray &ray,
                                         Random &rnd,
                                         float majorant,
                                         //output:
                                         vec3f &albedo,
                                         float &extinction)
{
  auto &lp = optixLaunchParams;

  float t=ray.tmin;

  while (1) {
    // In later chapters majorants will vary in space:
    if (majorant <= 0.f)
      break;

    t -= (logf(1.f - rnd()) / (majorant / lp.unitDistance));

    if (t > ray.tmax)
      break;

    vec3f P = ray.org+ray.dir*t;

    float value{0.f};
    if (!sampleVolume(lp.volume, P, value))
      continue;

    vec4f sample = postClassify(lp.transfunc, value);
    float u = rnd();
    if (sample.w >= u * majorant) {
      albedo = vec3f(sample.x,sample.y,sample.z);
      extinction = sample.w;
      break;
    }
  }

  return fminf(t,ray.tmax);
}

// ========================================================
// OptiX ICON geometry (only when using OWL!)
// ========================================================
#ifdef RTCORE
OPTIX_BOUNDS_PROGRAM(ICONCellBounds)(const void *geomData,
                                     owl::box3f &result, // mind the owl:: namespace!
                                     int leafID)
{
  const ICONGrid &self = *(const ICONGrid *)geomData;
  //result = ...[leafID];
  auto b = self.cells[leafID].getBounds();
  result = owl::box3f({b.lower.x,b.lower.y,b.lower.z},
                      {b.upper.x,b.upper.y,b.upper.z});
}

OPTIX_INTERSECT_PROGRAM(ICONCellIntersect)()
{
  const ICONGrid &self = owl::getProgramData<ICONGrid>();
  int leafID = optixGetPrimitiveIndex();
  owl::Ray ray(optixGetObjectRayOrigin(),
               optixGetObjectRayDirection(),
               optixGetRayTmin(),
               optixGetRayTmax());

  vec3f pos(ray.origin.x,ray.origin.y,ray.origin.z);
  float value{0.f};
  if (sample(self.cells[leafID],pos,value)) {
    if (optixReportIntersection(ray.tmin, 0)) {
      PRD &prd = owl::getPRD<PRD>();
      prd.value = value;
      prd.primID = leafID;
    }
  }
}

OPTIX_CLOSEST_HIT_PROGRAM(ICONCellClosestHit)()
{
  // empty
}

// CH used with triangle geom:
OPTIX_CLOSEST_HIT_PROGRAM(ICONTrianglesClosestHit)()
{
  PRD &prd = owl::getPRD<PRD>();
  prd.primID = optixGetPrimitiveIndex();
}
#endif

// ========================================================
// Sphere intersection, used for the makeshift
// traversal structure
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

inline __device__
bool traverseAccel(const Ray &ray, float &tnear, float &tfar, float &majorant) {
  auto &lp = optixLaunchParams;

  float t1,t2,t3,t4;
  bool s1 = intersectSphere(ray,lp.volume.accel.outerRadius,t1,t4);
  bool s2 = intersectSphere(ray,lp.volume.accel.innerRadius,t2,t3);
  if (!s1 && !s2) return false;
  if (t4 < ray.tmin) return false;
  // outer sphere hit, but inner was missed:
  if (s1 && !s2) {
    tnear = t1;
    tfar  = t4;
  }
  // inside front segment:
  else if (ray.tmin < t2) {
    tnear = t1;
    tfar  = t2;
  }
  // inside back segment:
  else { 
    tnear = t3;
    tfar  = t4;
  }
  majorant = 1.f;
  return true;
}

// ========================================================
// Ray gen prog (woodcock tracking, A+E, no accel)
// ========================================================
RAYGEN_PROGRAM(woodcockTrackingAE)()
{
  auto &lp = optixLaunchParams;
  const vec2i threadIndex = getLaunchIndex();
  const vec2i launchDim = getLaunchDims();
  const int pixelID = threadIndex.x + getLaunchDims().x * threadIndex.y;

  Random rnd(lp.accumID*launchDim.x*launchDim.y+(unsigned)threadIndex.x,
             (unsigned)threadIndex.y);

  Ray ray = generateRay(vec2f(threadIndex)+vec2f(.5f), rnd);

  float t0, t1;
  if (!boxTest(ray, lp.volume.bounds, t0, t1))
    return;

  ray.tmin = t0, ray.tmax = t1;

  const float majorant = 1.f;

  vec3f albedo = 0.f;
  float extinction = 0.f;

  float t = woodcockTracking(ray, rnd, majorant, albedo, extinction);

  vec3f color = albedo * lp.ambientColor * lp.ambientRadiance;
  float alpha = extinction > 0.f ? 1.f : 0.f;

  float accum = 1.f/(lp.accumID+1);
  lp.accumBuffer[pixelID] = lerp(vec4f(color,alpha), lp.accumBuffer[pixelID], accum);

  vec4f accumColor = lp.accumBuffer[pixelID];
  accumColor.r = linear_to_srgb(accumColor.r);
  accumColor.g = linear_to_srgb(accumColor.g);
  accumColor.b = linear_to_srgb(accumColor.b);
  lp.fbPointer[pixelID] = make_rgba(accumColor);
}


// ========================================================
// Ray gen prog (woodcock tracking, A+E, with naive accel)
// ========================================================
RAYGEN_PROGRAM(woodcockTrackingWithAccel)()
{
  auto &lp = optixLaunchParams;
  const vec2i threadIndex = getLaunchIndex();
  const vec2i launchDim = getLaunchDims();
  const int pixelID = threadIndex.x + getLaunchDims().x * threadIndex.y;

  Random rnd(lp.accumID*launchDim.x*launchDim.y+(unsigned)threadIndex.x,
             (unsigned)threadIndex.y);

  Ray ray = generateRay(vec2f(threadIndex)+vec2f(.5f), rnd);

  float t0, t1;
  if (!boxTest(ray, lp.volume.bounds, t0, t1))
    return;

  ray.tmin = t0, ray.tmax = t1;

  vec3f color{0.f};
  float alpha{0.f};

  if (lp.volume.accelMode == SPHERE_ACCEL_MODE) {
    float tnear, tfar;
    float majorant{1.f};
    while (traverseAccel(ray, tnear, tfar, majorant)) {
      vec3f albedo = 0.f;
      float extinction = 0.f;

      ray.tmin = fmaxf(ray.tmin,tnear);
      ray.tmax = tfar;
      float t = woodcockTracking(ray, rnd, majorant, albedo, extinction);
      if (t < tfar) {
        color = albedo * lp.ambientColor * lp.ambientRadiance;
        alpha = extinction > 0.f ? 1.f : 0.f;
        break;
      }
      // makeshift epsilon to avoid intersecting the same
      // spherical shell again (there are better ways to do this..)
      const float sceneEPS = lp.volume.accel.innerRadius*1e-3f;
      ray.tmin = tfar+sceneEPS;
    }
  } else {
    auto woodcockFunc = [&](const int leafID, float t0, float t1) {
      vec3f albedo = 0.f;
      float extinction = 0.f;
      const float majorant = lp.volume.gridAccel.maxOpacities[leafID];
      ray.tmin = t0;
      ray.tmax = t1;
      float t = woodcockTracking(ray, rnd, majorant, albedo, extinction);
      // if (debug()) {
      //   printf("%f:%f %f %f\n",t0,t1,majorant,t);
      // }
      if (t > t0 && t < t1) {
        color = albedo * lp.ambientColor * lp.ambientRadiance;
        alpha = extinction > 0.f ? 1.f : 0.f;
        return false;
      }
      return true; // traverse on
    };
    dda3(ray,lp.volume.gridAccel.dims,lp.volume.gridAccel.worldBounds,woodcockFunc);
  }

  float accum = 1.f/(lp.accumID+1);
  lp.accumBuffer[pixelID] = lerp(vec4f(color,alpha), lp.accumBuffer[pixelID], accum);

  vec4f accumColor = lp.accumBuffer[pixelID];
  accumColor.r = linear_to_srgb(accumColor.r);
  accumColor.g = linear_to_srgb(accumColor.g);
  accumColor.b = linear_to_srgb(accumColor.b);
  lp.fbPointer[pixelID] = make_rgba(accumColor);
}

} // namespace icon_rt



