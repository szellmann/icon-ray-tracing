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
// sdl3
# include <SDL3/SDL.h>
// ours
#include "vecmath.h"

namespace dvr_course {

using vecmath::vec2i;
using vecmath::vec4f;

typedef std::vector<vec4f> LUT;
typedef std::vector<int>   Histogram;

class AlphaEditor {
public:
  ~AlphaEditor();

  // Set SDL3 renderer
  void setSDL3Renderer(SDL_Renderer *renderer);

  //! Set a user-provided LUT that a copy is created from
  void setLookupTable(const LUT &rgba);

  //! Get an updated LUT that is a copied of the user-provided one
  LUT getUpdatedLookupTable();

  //! Optionally set a histogram that can be displayed instead of the LUT
  void setHistogram(const Histogram &hist);

  //! Set a zoom range to visually zoom into the LUT
  void setZoom(float min, float max);

  //! Indicates that the internal copy of the LUT has changed
  bool updated() const;

  //! Render with ImGui, open in new ImgGui window
  void show();

  //! Render with ImGui but w/o window
  void drawImmediate();

private:
  // Local LUT copy
  LUT rgbaLookupTable_;

  // User-provided LUT
  LUT userLookupTable_;

  // Local histogram with normalized data (optional)
  Histogram normalizedHistogram_;

  // User-provided histrogram
  Histogram userHistogram_;

  // Zoom min set by user
  float zoomMin_{0.f};

  // Zoom max set by user
  float zoomMax_{1.f};

  // Flag indicating that texture needs to be regenerated
  bool lutChanged_{false};

  // Flag indicating that texture needs to be regenerated
  bool histogramChanged_{false};

  // SDL3 renderer
  SDL_Renderer *sdl_renderer_{nullptr};

  // RGBA texture
  SDL_Texture *texture_{nullptr};

  // Drawing canvas size
  vec2i canvasSize_{ 300, 150 };

  // Mouse state for drawing
  struct MouseEvent {
    enum Type { PassiveMotion, Motion, Press, Release };
    enum Button { Left, Middle, Right, None };

    vec2i pos{ 0, 0 };
    int button{None};
    Type type{Motion};
  };

  // The last mouse event
  MouseEvent lastEvent_;

  // Drawing in progress
  bool drawing_ = false;

  // Resample the original LUT, result in rgbaLookupTable_
  void resampleOriginalLUT();

  // Raster LUT to image and upload with OpenGL
  void rasterTexture();

  // Generate normalized from user-provided histogram
  void normalizeHistogram();

  // Generate mouse event when mouse hovered over rect
  MouseEvent generateMouseEvent();

  // Handle mouse event
  void handleMouseEvent(MouseEvent const& event);
};

} // dvr_course


