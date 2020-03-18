cmake_minimum_required(VERSION 3.6.0 FATAL_ERROR)

message(STATUS "Start configuration of mptesnor")
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(mptensor
        VERSION 0.2.1
        LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/config)
if(CONFIG)
  message(STATUS "Loading configration: " ${PROJECT_SOURCE_DIR}/config/${CONFIG}.cmake)
  include(${PROJECT_SOURCE_DIR}/config/${CONFIG}.cmake)
endif(CONFIG)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

option(MPTENSOR_BUILD_LIBS_ONLY "Build only library" OFF)
option(BUILD_SHARED_LIBS "Build as shared library" OFF)
option(BUILD_DOCS "Build documents" OFF)
option(ENABLE_CXX11 "use c++11" ON)
option(ENABLE_MPI "use MPI and ScaLAPACK" ON)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH 1)

if(ENABLE_CXX11)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(ENABLED_CXX11 ON)
else()
  set(ENABLED_CXX11 OFF)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OpenMP_CXX_FOUND OR OPENMP_FOUND)

find_package(LAPACK)
if(NOT LAPACK_FOUND)
  message(STATUS "Could NOT find LAPACK library. Retry a search with C language.")
  enable_language(C)
  find_package(LAPACK REQUIRED)
endif()

if(ENABLE_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")

  find_package(ScaLAPACK REQUIRED)
else()
  add_definitions(-D_NO_MPI)
endif()

if(ENABLED_CXX11)
  add_definitions(-D_USE_RANDOM_CPP11)
else()
  find_package(Boost COMPONENTS random REQUIRED)
  include_directories(${Boost_INCLUDE_DIR})
  add_definitions(-D_USE_RANDOM_BOOST)
endif()

include_directories(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(src)

if(NOT MPTENSOR_BUILD_LIBS_ONLY)
  add_subdirectory(tests EXCLUDE_FROM_ALL)
  add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()

if(BUILD_DOCS)
  add_subdirectory(doc/doxygen)
endif()
