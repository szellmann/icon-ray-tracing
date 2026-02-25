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

#include "vecmath.h"

namespace dvr_course {

using namespace vecmath;

//=============================================================================
// camera class:
//=============================================================================
struct Camera
{
  void setAspect(float a) {
    aspect = a;
  }

  void setOrientation(const vec3f &origin,
                      const vec3f &poi,
                      const vec3f &up,
                      float fovy)
  {
    position = origin;
    upVector = up;
    this->fovy = fovy;
    frame.vz
      = (poi==origin)
      ? vec3f(0,0,1)
      : /* negative because we use NEGATIZE z axis */ - normalize(poi - origin);
    frame.vx = cross(up,frame.vz);
    if (dot(frame.vx,frame.vx) < 1e-8f)
      frame.vx = vec3f(0,1,0);
    else
      frame.vx = normalize(frame.vx);
    frame.vy = normalize(cross(frame.vz,frame.vx));
    distance = length(poi-origin);
    forceUpFrame();
  }

  void forceUpFrame()
  {
    // frame.vz remains unchanged
    if (fabsf(dot(frame.vz,upVector)) < 1e-6f)
      // looking along upvector; not much we can do here ...
      return;
    frame.vx = normalize(cross(upVector,frame.vz));
    frame.vy = normalize(cross(frame.vz,frame.vx));
  }

  vec3f getPosition() const {
    return position;
  }

  vec3f getPOI() const {
    return position-frame.vz*distance;
  }

  vec3f getUp() const {
    return upVector;
  }

  float getFovyInDegrees() const {
    return fovy/M_PI*180.f;
  }

  void getScreen(vec3f &lower_left, vec3f &horizontal, vec3f &vertical) const {
    float screen_height = 2.f*tanf(0.5f*fovy);
    vertical   = screen_height * frame.vy;
    horizontal = screen_height * aspect * frame.vx;
    lower_left
      =
      /* NEGATIVE z axis! */
      -frame.vz
      - 0.5f * vertical
      - 0.5f * horizontal;
  }

  void viewAll(const box3f &box) {
    vec3f up(0,1,0);
    float diagonal = length(box.size());
    float r = diagonal * 0.5f;
    vec3f eye = box.center() + vec3f(0, 0, r + r / std::atan(fovy));
    setOrientation(eye, box.center(), up, this->fovy);
  }

  vec3f position, upVector;
  float distance;
  float fovy{90.f*M_PI/180.f};
  float aspect{1.f};

  struct {
    vec3f vx, vy, vz;
  } frame;
};


//=============================================================================
// camera manip:
//=============================================================================
struct CameraManip
{
  enum MouseButton { Left, Middle, Right, None, };

  CameraManip() = default;

  CameraManip(Camera *cam, int w, int h)
    : camera(cam), vpWidth(w), vpHeight(h) {}

  bool handleMouseDown(int x, int y, MouseButton button) {
    if (!camera) return false;
    dragging = true;
    lastPos = {x,y};

    if (button == Left) {
      arcball.downPos = ballProject(x,y);
      arcball.downRotation = arcball.currRotation;
    }

    mouseButton = button;
    return true;
  }

  bool handleMouseUp(int x, int y, MouseButton /*button*/) {
    if (!camera) return false;
    dragging = false;
    mouseButton = None;
    return true;
  }

  bool handleMouseMove(int x, int y) {
    if (!camera || !dragging)
      return false;

    if (mouseButton == Left) {
      vec3f currPos = ballProject(x,y);
      arcball.currRotation
        = quatf::rotation(arcball.downPos, currPos) * arcball.downRotation;

      // update camera:
      mat4f rotmat = rotationMatrix(conjugate(arcball.currRotation));

      vec3f poi = camera->getPOI();

      vec4f eye4(0.f,0.f,camera->distance,1.f);
      eye4 = rotmat * eye4;
      vec3f eye(eye4.x,eye4.y,eye4.z);
      eye += poi;

      vec4f up4 = rotmat(1);
      vec3f up(up4.x,up4.y,up4.z);

      camera->setOrientation(eye, poi, up, camera->fovy);
    }
    else if (mouseButton == Right) {
      vec2i currPos{x,y};
      float dy = -float(lastPos.y - currPos.y) / vpHeight;
      float s = 2.f * camera->distance * dy;
      vec3f dir = normalize(camera->getPosition() - camera->getPOI());
      vec3f eye = camera->getPosition() - dir * s;
      camera->setOrientation(eye, camera->getPOI(), camera->getUp(), camera->fovy);
    }

    lastPos = {x,y};
    return true;
  }

  vec3f ballProject(int x, int y) {
    const float radius{1.f};
    vec3f v(0.f);
    v.x =  (x-0.5f*vpWidth ) / (radius*0.5f*vpWidth);
    v.y = -(y-0.5f*vpHeight) / (radius*0.5f*vpHeight);

    float d = v.x*v.x+v.y*v.y;
    if (d > 1.f) {
      float length = sqrtf(d);
      v.x /= length;
      v.y /= length;
    } else {
      v.z = sqrtf(1.f-d);
    }
    return v;
  }

  Camera *camera{nullptr};
  bool dragging{false};
  MouseButton mouseButton{None};
  vec2i lastPos{0,0};
  int vpWidth{0}, vpHeight{0};

  struct {
    vec3f downPos{0.f,0.f,0.f};
    quatf currRotation{quatf::identity()};
    quatf downRotation{quatf::identity()};
  } arcball;
};

} // namespace dvr_course



