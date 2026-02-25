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

#include <type_traits>
// ours
#include "vecmath.h"

#ifndef __both__
#define __both__ __host__ __device__
#endif

namespace dvr_course {

//=============================================================================
// Simple 1-D range class
//=============================================================================
template <typename I>
class range1d
{
public:
  static_assert(std::is_integral<I>::value, "Type must be integral.");

  __both__ range1d(I b, I e) : begin_(b) , end_(e)
  {}

  __both__ I&        begin()       { return begin_; }
  __both__ I const&  begin() const { return begin_; }
  __both__ I const& cbegin() const { return begin_; }

  __both__ I&        end()         { return end_; }
  __both__ I const&  end() const   { return end_; }
  __both__ I const& cend() const   { return end_; }

  __both__ I length() const {
      return end_ - begin_;
  }

private:
  I begin_;
  I end_;
};


//=============================================================================
// Simple 2-D range class
//=============================================================================
template <typename I>
class range2d
{
public:
  static_assert(std::is_integral<I>::value, "Type must be integral.");

  __both__ range2d(I rb, I re, I cb, I ce) : rows_(rb, re) , cols_(cb, ce)
  {}

  __both__ range1d<I>& rows()             { return rows_; }
  __both__ range1d<I> const& rows() const { return rows_; }

  __both__ range1d<I>& cols()             { return cols_; }
  __both__ range1d<I> const& cols() const { return cols_; }

private:
  range1d<I> rows_;
  range1d<I> cols_;
};


//=============================================================================
// Simple 3-D range class
//=============================================================================
template <typename I>
class range3d
{
public:
  static_assert(std::is_integral<I>::value, "Type must be integral.");

  __both__ range3d(I rb, I re, I cb, I ce, I sb, I se)
      : rows_(rb, re) , cols_(cb, ce), slices_(sb, se)
  {}

  __both__ range1d<I>& rows()               { return rows_; }
  __both__ range1d<I> const& rows() const   { return rows_; }

  __both__ range1d<I>& cols()               { return cols_; }
  __both__ range1d<I> const& cols() const   { return cols_; }

  __both__ range1d<I>& slices()             { return slices_; }
  __both__ range1d<I> const& slices() const { return slices_; }

private:
  range1d<I> rows_;
  range1d<I> cols_;
  range1d<I> slices_;
};


//=============================================================================
// 1-D tiled range class
//=============================================================================
template <typename I>
class tiled_range1d
{
public:
  static_assert(std::is_integral<I>::value, "Type must be integral.");

  __both__ tiled_range1d(I b, I e, I ts)
      : begin_(b), end_(e), tile_size_(ts)
  {}

  __both__ I&        begin()           { return begin_; }
  __both__ I const&  begin()     const { return begin_; }
  __both__ I const& cbegin()     const { return begin_; }

  __both__ I&        end()             { return end_; }
  __both__ I const&  end()       const { return end_; }
  __both__ I const& cend()       const { return end_; }

  __both__ I&        tile_size()       { return tile_size_; }
  __both__ I const&  tile_size() const { return tile_size_; }

  __both__ I length() const {
      return end_ - begin_;
  }

private:
  I begin_;
  I end_;
  I tile_size_;
};


//=============================================================================
// 2-D tiled range class
//=============================================================================
template <typename I>
class tiled_range2d
{
public:
  static_assert(std::is_integral<I>::value, "Type must be integral.");

  __both__ tiled_range2d(I rb, I re, I rts, I cb, I ce, I cts)
      : rows_(rb, re, rts), cols_(cb, ce, cts)
  {}

  __both__ tiled_range1d<I>& rows()             { return rows_; }
  __both__ tiled_range1d<I> const& rows() const { return rows_; }

  __both__ tiled_range1d<I>& cols()             { return cols_; }
  __both__ tiled_range1d<I> const& cols() const { return cols_; }

private:
  tiled_range1d<I> rows_;
  tiled_range1d<I> cols_;
};


//=============================================================================
// 3-D tiled range class
//=============================================================================
template <typename I>
class tiled_range3d
{
public:
  static_assert(std::is_integral<I>::value, "Type must be integral.");

  __both__ tiled_range3d(I rb, I re, I rts, I cb, I ce, I cts, I sb, I se, I sts)
      : rows_(rb, re, rts), cols_(cb, ce, cts), slices_(sb, se, sts)
  {}

  __both__ tiled_range1d<I>& rows()               { return rows_; }
  __both__ tiled_range1d<I> const& rows() const   { return rows_; }

  __both__ tiled_range1d<I>& cols()               { return cols_; }
  __both__ tiled_range1d<I> const& cols() const   { return cols_; }

  __both__ tiled_range1d<I>& slices()             { return slices_; }
  __both__ tiled_range1d<I> const& slices() const { return slices_; }

private:
  tiled_range1d<I> rows_;
  tiled_range1d<I> cols_;
  tiled_range1d<I> slices_;
};

} // dvr_course


