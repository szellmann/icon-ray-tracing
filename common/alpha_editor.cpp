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

// std
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
// imgui
#include <imgui.h>
// ours
#define GL_SILENCE_DEPRECATION // for macOS
#include "alpha_editor.h"
#include "vecmath.h"
#include "dvr_course-common.h"

using namespace vecmath;

static void enableBlendCB(ImDrawList const*, ImDrawCmd const*)
{
  //glEnable(GL_BLEND);
}

static void disableBlendCB(ImDrawList const*, ImDrawCmd const*)
{
  //glDisable(GL_BLEND);
}

namespace dvr_course {
  AlphaEditor::~AlphaEditor()
  {
    if (texture_) {
      SDL_DestroyTexture(texture_);
    }
  }

  void AlphaEditor::setSDL3Renderer(SDL_Renderer *renderer)
  {
    sdl_renderer_ = renderer;
  }

  void AlphaEditor::setLookupTable(const LUT &rgba)
  {
    lutChanged_ = true;
    userLookupTable_ = rgba;
  }

  LUT AlphaEditor::getUpdatedLookupTable()
  {
    if (!userLookupTable_.empty() && rgbaLookupTable_.empty())
      resampleOriginalLUT();

    return rgbaLookupTable_;
  }

  void AlphaEditor::setHistogram(const Histogram &hist)
  {
    histogramChanged_ = true;

    userHistogram_ = hist;
  }

  void AlphaEditor::setZoom(float min, float max)
  {
    zoomMin_ = min;
    zoomMax_ = max;
  }

  bool AlphaEditor::updated() const
  {
      return lutChanged_;
  }

  void AlphaEditor::show()
  {
    if (userLookupTable_.empty())
      return;

    ImGui::Begin("AlphaEditor");

    drawImmediate();

    ImGui::End();
  }

  void AlphaEditor::drawImmediate()
  {
    if (userLookupTable_.empty())
      return;

    rasterTexture();

    //ImGui::GetWindowDrawList()->AddCallback(disableBlendCB, nullptr);
    ImVec2 old_padding = ImGui::GetStyle().FramePadding;
    ImGui::GetStyle().FramePadding = ImVec2(0,0);
    ImGui::ImageButton("##", (ImU64)texture_, ImVec2(canvasSize_.x,canvasSize_.y));
    ImGui::GetStyle().FramePadding = old_padding;

    MouseEvent event = generateMouseEvent();
    handleMouseEvent(event);

    //ImGui::GetWindowDrawList()->AddCallback(enableBlendCB, nullptr);
  }

  void AlphaEditor::rasterTexture()
  {
    if (!sdl_renderer_) {
      fprintf(stderr, "%s", "SDL3 renderer not set!\n");
      abort();
    }

    if (!lutChanged_ && !histogramChanged_)
        return;

    if (histogramChanged_)
      normalizeHistogram();

    if (!texture_) {
      texture_ = SDL_CreateTexture(sdl_renderer_,
          SDL_PIXELFORMAT_RGBA32,
          SDL_TEXTUREACCESS_STREAMING,
          canvasSize_.x,
          canvasSize_.y);
    }

    int userDims = (int)userLookupTable_.size();

    if (userDims >= 1) {
      if (rgbaLookupTable_.empty())
        resampleOriginalLUT();

      // Updated colors
      const vec4f *colors = rgbaLookupTable_.data();


      // Blend on the CPU (TODO: figure out how textures can be
      // blended with ImGui..)
      std::vector<uint32_t> rgba(canvasSize_.x * canvasSize_.y);
      int actualDims = (int)rgbaLookupTable_.size();
      for (int y=0; y<canvasSize_.y; ++y) {
        for (int x=0; x<canvasSize_.x; ++x) {
          float indexf = x / (float)(canvasSize_.x - 1);
          indexf *= zoomMax_ - zoomMin_;
          indexf += zoomMin_;
          indexf *= actualDims - 1;
          int xx = (int)indexf;

          vec3f rgb(colors[xx].x, colors[xx].y, colors[xx].z);
          float alpha = colors[xx].w;

          if (!normalizedHistogram_.empty()) {
            float binf = x / (float)(canvasSize_.x - 1);
            binf *= zoomMax_ - zoomMin_;
            binf += zoomMin_;
            binf *= normalizedHistogram_.size() - 1;
            int bin = (int)binf;

            int yy = canvasSize_.y - y - 1;
            if (yy <= normalizedHistogram_[bin]) {
              float lum = .3f * rgb.x + .59f * rgb.y + .11f * rgb.z;
              rgb = { 1.f - lum, 1.f - lum, 1.f - lum };
            }
          }

          float grey = .9f;
          float a = ((canvasSize_.y - y - 1) / (float)canvasSize_.y) <= alpha ? .6f : 0.f;

          vec4f clr;
          clr.x = (1-a)*rgb.x + a*grey;
          clr.y = (1-a)*rgb.y + a*grey;
          clr.z = (1-a)*rgb.z + a*grey;
          clr.w = 1.f;
          rgba[y*canvasSize_.x+x] = make_rgba(clr);
        }
      }

      SDL_UpdateTexture(texture_,
          nullptr,
          rgba.data(), // TODO: convert to uint
          canvasSize_.x * sizeof(uint32_t));
    } else {
      fprintf(stderr, "%s", "No user LUT provided!\n");
      abort();
    }

    lutChanged_ = false;
  }

  void AlphaEditor::resampleOriginalLUT()
  {
    rgbaLookupTable_.resize(canvasSize_.x);
    dvr_course::resampleLUT(rgbaLookupTable_, userLookupTable_);
  }

  void AlphaEditor::normalizeHistogram()
  {
    assert(!userHistogram_.empty());

    normalizedHistogram_.resize(userHistogram_.size());

    int maxBinCount = 0;

    for (size_t i=0; i<userHistogram_.size(); ++i) {
      maxBinCount = std::max(maxBinCount, userHistogram_[i]);
    }

    if (maxBinCount == 0) {
      std::fill(normalizedHistogram_.begin(), normalizedHistogram_.end(), 0);
    } else {
      for (size_t i=0; i<userHistogram_.size(); ++i) {
        float countf = true // (TODO)
            ? logf((float)userHistogram_[i]) / logf((float)maxBinCount)
            : userHistogram_[i] / (float)maxBinCount;

        normalizedHistogram_[i] = (size_t)(countf * canvasSize_.y);
      }
    }

    histogramChanged_ = false;
  }

  AlphaEditor::MouseEvent AlphaEditor::generateMouseEvent()
  {
    MouseEvent event;

    int x = ImGui::GetIO().MousePos.x - ImGui::GetCursorScreenPos().x;
    int y = ImGui::GetCursorScreenPos().y - ImGui::GetIO().MousePos.y - 1;
   
    event.pos = { x, y };
    event.button = ImGui::GetIO().MouseDown[0] ? MouseEvent::Left :
                   ImGui::GetIO().MouseDown[1] ? MouseEvent::Middle :
                   ImGui::GetIO().MouseDown[2] ? MouseEvent::Right:
                                                 MouseEvent::None;
    // TODO: handle the unlikely case that the down button is not
    // the same as the one from lastEvent_. This could happen as
    // the mouse events are tied to the rendering frame rate
    if (event.button == MouseEvent::None && lastEvent_.button == MouseEvent::None)
      event.type = MouseEvent::PassiveMotion;
    else if (event.button != MouseEvent::None && lastEvent_.button != MouseEvent::None)
      event.type = MouseEvent::Motion;
    else if (event.button != MouseEvent::None && lastEvent_.button == MouseEvent::None)
      event.type = MouseEvent::Press;
    else
      event.type = MouseEvent::Release;

    return event;
  }

  void AlphaEditor::handleMouseEvent(const AlphaEditor::MouseEvent &event)
  {
    bool hovered = ImGui::IsItemHovered()
        && event.pos.x >= 0 && event.pos.x < canvasSize_.x
        && event.pos.y >= 0 && event.pos.y < canvasSize_.y;

    if (event.type == MouseEvent::PassiveMotion || event.type == MouseEvent::Release)
      drawing_ = false;

    if (drawing_ || (event.type == MouseEvent::Press && hovered && event.button == MouseEvent::Left)) {
      vec4f *updated = rgbaLookupTable_.data();

      int actualDims = (int)rgbaLookupTable_.size();

      // Allow for drawing even when we're slightly outside
      // (i.e. not hovering) the drawing area
      int thisX = clamp(event.pos.x, 0, canvasSize_.x - 1);
      int thisY = clamp(event.pos.y, 0, canvasSize_.y - 1);
      int lastX = clamp(lastEvent_.pos.x, 0, canvasSize_.x - 1);

      auto zoom = [=](int x) {
        float indexf = x / (float)(canvasSize_.x - 1);
        indexf *= zoomMax_ - zoomMin_;
        indexf += zoomMin_;
        indexf *= actualDims - 1;
        return (int)indexf;
      };

      updated[zoom(thisX)].w = thisY / (float)(canvasSize_.y-1);

      // Also set the alphas that were potentially skipped b/c
      // the mouse movement was faster than the rendering frame
      // rate
      if (lastEvent_.button == MouseEvent::Left && std::abs(lastX-thisX) > 1) {
        float alpha1;
        float alpha2;
        if (lastX < thisX) {
          alpha1 = updated[zoom(lastX)].w;
          alpha2 = updated[zoom(thisX)].w;
        } else {
          alpha1 = updated[zoom(thisX)].w;
          alpha2 = updated[zoom(lastX)].w;
        }

        int inc = lastEvent_.pos.x < event.pos.x ? 1 : -1;

        for (int x=zoom(lastX)+inc; x!=zoom(thisX); x+=inc) {
          float frac = (zoom(thisX) - x) / (float)std::abs(zoom(thisX) - zoom(lastX));
          updated[x].w = lerp(alpha1, alpha2, frac);
        }
      }

      lutChanged_ = true;
      drawing_ = true;
    }

    lastEvent_ = event;
  }
} // dvr_course


