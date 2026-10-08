# GoogleTest: prefer an installed package, otherwise fetch a pinned release.
find_package(GTest 1.14 CONFIG QUIET)
if(GTest_FOUND)
  message(STATUS "GoogleTest: found ${GTest_VERSION} (${GTest_DIR})")
else()
  include(FetchContent)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.18.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
  message(STATUS "GoogleTest: using FetchContent (v1.18.0)")
endif()

# Shared main and helpers for all gtest-based tests.
add_library(mptensor_gtest_main STATIC common/gtest_mpi_main.cc)
target_link_libraries(mptensor_gtest_main PUBLIC mptensor GTest::gtest)
target_include_directories(mptensor_gtest_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

# mptensor_add_test(<name> <target>
#                   [WORKING_DIRECTORY <dir>]
#                   [FIXTURES_SETUP <fixture>...]
#                   [FIXTURES_REQUIRED <fixture>...])
# Registers <target> as <name>_mpi000N (N = 1..4) with MPI, or <name>_serial.
function(mptensor_add_test name target)
  cmake_parse_arguments(ARG "" "WORKING_DIRECTORY" "FIXTURES_SETUP;FIXTURES_REQUIRED" ${ARGN})
  if(NOT ARG_WORKING_DIRECTORY)
    set(ARG_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
  set(test_names)
  if(ENABLE_MPI)
    foreach(np RANGE 1 4)
      set(test_name ${name}_mpi000${np})
      add_test(NAME ${test_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:${target}> ${MPIEXEC_POSTFLAGS}
        WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY})
      list(APPEND test_names ${test_name})
    endforeach()
  else()
    set(test_name ${name}_serial)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${target}>
      WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY})
    list(APPEND test_names ${test_name})
  endif()
  # A hang (e.g. ranks diverging before a collective call) must fail, not block.
  set_tests_properties(${test_names} PROPERTIES TIMEOUT 600)
  if(ARG_FIXTURES_SETUP)
    set_tests_properties(${test_names} PROPERTIES FIXTURES_SETUP "${ARG_FIXTURES_SETUP}")
  endif()
  if(ARG_FIXTURES_REQUIRED)
    set_tests_properties(${test_names} PROPERTIES FIXTURES_REQUIRED "${ARG_FIXTURES_REQUIRED}")
  endif()
endfunction()

add_subdirectory(tensor)
add_subdirectory(file_io)

if(BUILD_LEGACY_TESTS)
  add_subdirectory(legacy)
endif()
