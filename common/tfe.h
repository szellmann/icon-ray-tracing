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

#include "alpha_editor.h"
#include "transfunc.h"

namespace dvr_course {

class TFE : private AlphaEditor
{
 public:
  void init(const Transfunc &transfunc);
  bool drawImmediate();

  // Set SDL3 renderer
  void setSDL3Renderer(SDL_Renderer *renderer)
  { AlphaEditor::setSDL3Renderer(renderer); }

  // general update:
  bool updated() const;

  // RGBA LUT updated:
  bool lutUpdated() const
  { return AlphaEditor::updated(); }

  // value range updated:
  bool rangeUpdated() const
  { return rangeUpdated_; }

  // rel-domain updated:
  bool domainUpdated() const
  { return domainUpdated_; }

  // opacity scale updated:
  bool scaleUpdated() const
  { return scaleUpdated_; }

  LUT getLUT()
  { return AlphaEditor::getUpdatedLookupTable(); }

  box1f getRange();
  box1f getRelDomain();
  float getOpacityScale();

 private:
  box1f range;
  box1f relDomain;
  float opacityScale;

  bool rangeUpdated_{false};
  bool domainUpdated_{false};
  bool scaleUpdated_{false};
};

} // dvr_course


