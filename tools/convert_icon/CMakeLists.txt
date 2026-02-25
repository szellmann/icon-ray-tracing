# Copyright 2025-2026 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(convert_icon LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(netCDF COMPONENTS netcdf REQUIRED)
add_executable(${PROJECT_NAME} convert_icon.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE netCDF::netcdf)

find_package(umesh)

if(TARGET umesh::umesh)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_UMESH)
  target_link_libraries(${PROJECT_NAME} PRIVATE umesh::umesh)
endif()
