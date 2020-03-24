cmake_minimum_required(VERSION 3.6.0 FATAL_ERROR)
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

option(ENABLE_MPI "Use MPI and ScaLAPACK" ON)
option(BUILD_SHARED_LIBS "Build as shared library" OFF)
option(BUILD_DOCS "Build documents" OFF)
option(BUILD_TESTS "Build tests" OFF)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(IS_SUB_PROJECT OFF)
else()
  set(IS_SUB_PROJECT ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
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

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif()

find_package(LAPACK)
if(NOT LAPACK_FOUND)
  message(STATUS "Could NOT find LAPACK library. Retry a search with C language.")
  enable_language(C)
  find_package(LAPACK REQUIRED)
endif()

if(ENABLE_MPI)
  find_package(MPI REQUIRED)
  if(NOT TARGET MPI::MPI_CXX)
    string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
  endif()
  find_package(ScaLAPACK REQUIRED)
else()
  add_definitions(-D_NO_MPI)
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
