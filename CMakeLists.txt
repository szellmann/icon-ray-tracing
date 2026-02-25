# Copyright 2025-2026 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(dvr-course-cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(common/cmake/common.cmake)

add_subdirectory(common)
add_subdirectory(icon_rt)

# cmdline tools used for data wrangling, etc.
# we don't usually need these to run the examples
option(BUILD_CMDLINE_TOOLS "Build with cmdline tools" OFF)
if(BUILD_CMDLINE_TOOLS)
  add_subdirectory(tools/convert_icon)
endif()
