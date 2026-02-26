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
#include <functional>
#include <memory>
#include <string>
// ours
#include "camera.h"
#include "fb.h"
#include "transfunc.h"

#ifndef RTCORE
# define DECL_LAUNCH_PARAMS(T) T optixLaunchParams;
# define SET_LAUNCH_PARAMS(p) optixLaunchParams = (p);
#else
# define DECL_LAUNCH_PARAMS(T)
# define SET_LAUNCH_PARAMS(p)
#endif

typedef void *RawPointer;

struct _OWLContext;
struct _OWLModule;
struct _OWLVarDecl;
typedef _OWLContext *OWLContext;
typedef struct _OWLLaunchParams *OWLLaunchParams, *OWLParams, *OWLGlobals;
typedef _OWLModule *OWLModule;
typedef _OWLVarDecl OWLVarDecl;

typedef unsigned long long OptixTraversableHandle;

// ========================================================
// Common render pipeline class for DVR
// ========================================================
namespace dvr_course {

struct Pipeline {

  Pipeline(std::string name = "dvr-course-cpp");
  Pipeline(int argc, char *argv[], std::string name = "dvr-course-cpp");
  ~Pipeline();

#ifdef RTCORE
  // for use with RTCORE (load from module)

  //   ray-gen
  void setRayGen(const char *name);
  void setRayGen(const char *ptxCode, const char *name);

  // lp-decl
  void setLaunchParamsDecl(OWLVarDecl *decl, size_t sizeOfLaunchParamsStruct);

  // get OWL context
  OWLContext owlContext();

  // get OWL module
  OWLModule owlModule();

  // get OWL params
  OWLParams owlLaunchParams();
#else
  // for use with non-RTCORE (set as function pointer)

  //   ray-gen
  void setRayGen(const std::function<void()> &f)
  { func = f; }

  std::function<void()> func;
#endif

  //   launch-params
#define DECL_LAUNCH_PARM_FUNC(T) T &launchParam(std::string name, T &value);

  DECL_LAUNCH_PARM_FUNC(bool)
  DECL_LAUNCH_PARM_FUNC(int)
  DECL_LAUNCH_PARM_FUNC(float)
  DECL_LAUNCH_PARM_FUNC(vec2f)
  DECL_LAUNCH_PARM_FUNC(vec3f)
  DECL_LAUNCH_PARM_FUNC(vec4f)
  DECL_LAUNCH_PARM_FUNC(box1f)
  DECL_LAUNCH_PARM_FUNC(box3f)
  DECL_LAUNCH_PARM_FUNC(RawPointer)
#ifdef RTCORE
  DECL_LAUNCH_PARM_FUNC(OptixTraversableHandle)
#endif

  // Frame
  void setFrame(Frame *f);
  Frame *fb{nullptr};

  int frameID{0};

  // Camera
  void setCamera(Camera *cam);
  Camera *camera{nullptr};

  // Transfunc
  void setTransfunc(Transfunc *tf, int index=0);
  Transfunc *getTransfunc(int index=0) const;
  bool transfuncValid(int index=0) const;

  // UI params
  void uiParam(std::string name, bool *b);
  void uiParam(std::string name, float *f, float minf, float maxf);
  void uiParam(std::string name, const std::vector<std::string> &options, int *o);

  // Interface
  bool isValid() const { return fb != nullptr && camera != nullptr; }
  bool isRunning();
  void launch();
  void present() const;
  void resetAccumulation();

  // Events
  typedef std::function<void(char)> KeyDownHandler;
  void setKeyDownHandler(KeyDownHandler kdh);

 private:

  struct Impl;
  std::unique_ptr<Impl> impl;

  bool running{false};
};

} // dvr_course



