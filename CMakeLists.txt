cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)

message(STATUS "CMake version: ${CMAKE_VERSION}")

project(mptensor NONE)
set(mptensor_version "v0.2")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/config)
if(CONFIG)
  message(STATUS "Loading configration: " ${PROJECT_SOURCE_DIR}/config/${CONFIG}.cmake)
  include(${PROJECT_SOURCE_DIR}/config/${CONFIG}.cmake)
endif(CONFIG)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})


option(BUILD_SHARED_LIBS "Build as shared library" OFF)
option(BUILD_DOCS "Build documents" OFF)
option(ENABLE_CXX11 "use c++11" ON)
option(ENABLE_MPI "use MPI and ScaLAPACK" ON)

enable_language(C CXX)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH 1)


set(ENABLED_CXX11 OFF)
if(ENABLE_CXX11)
  include(${PROJECT_SOURCE_DIR}/config/cxx11.cmake)
  cxx11()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OpenMP_CXX_FOUND OR OPENMP_FOUND)

find_package(LAPACK REQUIRED)

if(ENABLE_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")

  find_package(ScaLAPACK REQUIRED)
  string(REGEX REPLACE ";" " " SCAPALACK_LIBS "${SCAPALACK_LIBRARIES}")
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

if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  configure_file(${PROJECT_SOURCE_DIR}/config/Doxyfile.in ${PROJECT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile)
  install(DIRECTORY doxygen_docs/html DESTINATION share/mptensor/${mptensor_version})
endif()

include_directories(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
