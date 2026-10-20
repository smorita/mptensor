cmake_minimum_required(VERSION 3.16...3.31)

message(STATUS "Start configuration of mptesnor")
message(STATUS "CMake version: ${CMAKE_VERSION}")

set(VERSION_REGEX "#define[ \t]+MPTENSOR_VERSION_STRING[ \t]+\"(.+)\"")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/mptensor/version.hpp" VERSION_STRING
     REGEX ${VERSION_REGEX})
string(REGEX REPLACE ${VERSION_REGEX} "\\1" VERSION_STRING "${VERSION_STRING}")
message(STATUS "Start configuration of mptesnor (v${VERSION_STRING})")

project(mptensor
        VERSION ${VERSION_STRING}
        LANGUAGES CXX
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(IS_SUB_PROJECT OFF)
else()
  set(IS_SUB_PROJECT ON)
endif()

# ENABLE_MPI is the old name of MPTENSOR_ENABLE_MPI.
if(DEFINED ENABLE_MPI AND NOT DEFINED MPTENSOR_ENABLE_MPI)
  message(DEPRECATION "ENABLE_MPI is deprecated; use MPTENSOR_ENABLE_MPI instead.")
  set(MPTENSOR_ENABLE_MPI ${ENABLE_MPI} CACHE BOOL "Use MPI and ScaLAPACK")
endif()
option(MPTENSOR_ENABLE_MPI "Use MPI and ScaLAPACK" ON)

if(IS_SUB_PROJECT)
  set(_mptensor_install_default OFF)
else()
  set(_mptensor_install_default ON)
endif()
option(MPTENSOR_INSTALL "Install mptensor (default: OFF when used as a subproject)"
       ${_mptensor_install_default})

option(BUILD_SHARED_LIBS "Build as shared library" OFF)
option(BUILD_DOCS "Build documents" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_LEGACY_TESTS "Build legacy tests" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT IS_SUB_PROJECT)
  if(BUILD_TESTS)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Type of build" FORCE)
  else()
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
  endif()
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

find_package(LAPACK)
if(NOT LAPACK_FOUND)
  message(STATUS "Could NOT find LAPACK library. Retry a search with C language.")
  enable_language(C)
  find_package(LAPACK REQUIRED)
endif()

if(MPTENSOR_ENABLE_MPI)
  find_package(MPI REQUIRED)
  find_package(ScaLAPACK REQUIRED)
endif()

add_subdirectory(src)

if(NOT IS_SUB_PROJECT)
  add_subdirectory(examples EXCLUDE_FROM_ALL)

  if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
  endif()

  if(BUILD_DOCS)
    add_subdirectory(doc/doxygen)
  endif()

  include(FeatureSummary)
  feature_summary(WHAT ENABLED_FEATURES DISABLED_FEATURES PACKAGES_FOUND)
endif()
