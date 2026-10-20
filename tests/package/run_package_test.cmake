# Build and run tests/package/consumer against mptensor.
#
#   cmake -DMODE=install|subdir -DSOURCE_DIR=... -DBUILD_DIR=... -DWORK_DIR=...
#         -DGENERATOR=... -DCXX=... -DC=... -DBUILD_TYPE=... -DENABLE_MPI=ON|OFF
#         -DEXPECTED_VERSION=... [-DMPIEXEC=... -DNUMPROC_FLAG=... -DPREFLAGS=...]
#         -P run_package_test.cmake
#
# MODE=install: install the already-built mptensor from BUILD_DIR into a
#   temporary prefix and use it through find_package(mptensor).
# MODE=subdir: build mptensor from SOURCE_DIR with add_subdirectory, and check
#   that installing the consumer does not install mptensor itself.

function(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    string(REPLACE ";" " " cmd "${ARGN}")
    message(FATAL_ERROR "Command failed (${rc}): ${cmd}")
  endif()
endfunction()

file(REMOVE_RECURSE "${WORK_DIR}")
set(consumer_src "${SOURCE_DIR}/tests/package/consumer")
set(consumer_build "${WORK_DIR}/build")
set(configure_args -S "${consumer_src}" -B "${consumer_build}" -G "${GENERATOR}"
    -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_BUILD_TYPE=${BUILD_TYPE})
if(C)
  list(APPEND configure_args -DCMAKE_C_COMPILER=${C})
endif()

if(MODE STREQUAL "install")
  run(${CMAKE_COMMAND} --install "${BUILD_DIR}" --prefix "${WORK_DIR}/prefix")
  run(${CMAKE_COMMAND} ${configure_args}
      -DCMAKE_PREFIX_PATH=${WORK_DIR}/prefix
      -DEXPECTED_VERSION=${EXPECTED_VERSION})
elseif(MODE STREQUAL "subdir")
  run(${CMAKE_COMMAND} ${configure_args}
      -DMPTENSOR_SOURCE_DIR=${SOURCE_DIR}
      -DMPTENSOR_ENABLE_MPI=${ENABLE_MPI})
else()
  message(FATAL_ERROR "Unknown MODE: ${MODE}")
endif()

run(${CMAKE_COMMAND} --build "${consumer_build}")

if(ENABLE_MPI)
  separate_arguments(preflags UNIX_COMMAND "${PREFLAGS}")
  run(${MPIEXEC} ${NUMPROC_FLAG} 2 ${preflags} "${consumer_build}/consumer")
else()
  run("${consumer_build}/consumer")
endif()

if(MODE STREQUAL "subdir")
  set(prefix "${WORK_DIR}/prefix-consumer")
  run(${CMAKE_COMMAND} --install "${consumer_build}" --prefix "${prefix}")
  if(EXISTS "${prefix}/include/mptensor" OR EXISTS "${prefix}/lib/cmake/mptensor")
    message(FATAL_ERROR "mptensor was installed although it is a subproject")
  endif()
endif()
