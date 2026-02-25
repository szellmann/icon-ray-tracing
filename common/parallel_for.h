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

#include "range.h"
#include "thread_pool.h"

namespace dvr_course {

inline int div_up(int a, int b) {
  return (a+b-1)/b;
}

//=============================================================================
// parallel_for
//=============================================================================
template <typename I, typename Func>
void parallel_for(thread_pool& pool, range1d<I> const& range, Func const& func) {
  I len = range.length();
  I tile_size = div_up(len, static_cast<I>(pool.num_threads));
  I num_tiles = div_up(len, tile_size);

  pool.run([=](long tile_index) {
    I first = static_cast<I>(tile_index) * tile_size;
    I last = std::min(first + tile_size, len);

    for (I i = first; i != last; ++i) {
      func(i);
    }
  }, static_cast<long>(num_tiles));
}

template <typename I, typename Func>
void parallel_for(thread_pool& pool, tiled_range1d<I> const& range, Func const& func) {
  I beg = range.begin();
  I len = range.length();
  I tile_size = range.tile_size();
  I num_tiles = div_up(len, tile_size);

  pool.run([=](long tile_index) {
    I first = static_cast<I>(tile_index) * tile_size + beg;
    I last = std::min(first + tile_size, beg + len);

    func(range1d<I>(first, last));
  }, static_cast<long>(num_tiles));
}

template <typename I, typename Func>
void parallel_for(thread_pool& pool, tiled_range2d<I> const& range, Func const& func) {
  I first_row = range.rows().begin();
  I first_col  = range.cols().begin();
  I width = range.rows().length();
  I height = range.cols().length();
  I tile_width = range.rows().tile_size();
  I tile_height = range.cols().tile_size();
  I num_tiles_x = div_up(width, tile_width);
  I num_tiles_y = div_up(height, tile_height);

  pool.run([=](long tile_index) {
    I first_x = (tile_index % num_tiles_x) * tile_width + first_row;
    I last_x = std::min(first_x + tile_width, first_row + width);

    I first_y = (tile_index / num_tiles_x) * tile_height + first_col;
    I last_y = std::min(first_y + tile_height, first_col + height);

    func(range2d<I>(first_x, last_x, first_y, last_y));
  }, static_cast<long>(num_tiles_x * num_tiles_y));
}

} // dvr_course


