cmake_minimum_required(VERSION 3.6...3.17)

message(STATUS "Start configuration of mptesnor")
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(mptensor
        VERSION 0.3.0
        LANGUAGES CXX)

option(MPTENSOR_BUILD_LIBS_ONLY "Build only library" OFF)
option(BUILD_SHARED_LIBS "Build as shared library" OFF)
option(BUILD_DOCS "Build documents" OFF)
option(ENABLE_MPI "Use MPI and ScaLAPACK" ON)
option(ENABLE_TEST "Enable tests" OFF)

if(ENABLE_TEST)
  enable_testing()
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Type of build" FORCE)
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/config)
if(CONFIG)
  message(STATUS "Loading configration: " ${PROJECT_SOURCE_DIR}/config/${CONFIG}.cmake)
  include(${PROJECT_SOURCE_DIR}/config/${CONFIG}.cmake)
endif(CONFIG)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

add_subdirectory(src)

if(NOT MPTENSOR_BUILD_LIBS_ONLY)
  add_subdirectory(examples EXCLUDE_FROM_ALL)
  if(ENABLE_TEST)
    add_subdirectory(tests)
  else()
    add_subdirectory(tests EXCLUDE_FROM_ALL)
  endif()
endif()

if(BUILD_DOCS)
  add_subdirectory(doc/doxygen)
endif()
