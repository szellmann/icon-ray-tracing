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

#include <imgui.h>
#include "tfe.h"

namespace dvr_course {

void TFE::init(const Transfunc &transfunc) {
  AlphaEditor::setLookupTable(transfunc.getLUT());
  range = transfunc.valueRange;
  relDomain = transfunc.relRange;
  opacityScale = transfunc.opacity;
}

void TFE::drawImmediate() {
  AlphaEditor::drawImmediate();

  if (ImGui::DragFloatRange2("Range", &range.lower, &range.upper,
                             fmaxf(range.size()/100.f, 0.0001f))) {
    rangeUpdated_ = true;
  }

  if (ImGui::DragFloatRange2("Rel Domain", &relDomain.lower, &relDomain.upper, 1.f)) {
    domainUpdated_ = true;
  }

  if (ImGui::DragFloat("Opacity", &opacityScale, 1.f)) {
    scaleUpdated_ = true;
  }

  // if (ImGui::Button("Save")) {
  //   saveToFile(saveFilename.c_str());
  // }
}

bool TFE::updated() const {
  return AlphaEditor::updated() || rangeUpdated_ || domainUpdated_ || scaleUpdated_;
}

box1f TFE::getRange() {
  rangeUpdated_ = false;
  return range;
}

box1f TFE::getRelDomain() {
  domainUpdated_ = false;
  return relDomain;
}

float TFE::getOpacityScale() {
  scaleUpdated_ = false;
  return opacityScale;
}

} // namespace dvr_course



