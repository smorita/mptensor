# Check that downstream projects can use mptensor through
# find_package(mptensor) and through add_subdirectory().
foreach(mode install subdir)
  add_test(NAME package/${mode}
    COMMAND ${CMAKE_COMMAND}
            -DMODE=${mode}
            -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DBUILD_DIR=${PROJECT_BINARY_DIR}
            -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/${mode}
            -DGENERATOR=${CMAKE_GENERATOR}
            -DCXX=${CMAKE_CXX_COMPILER}
            -DC=${CMAKE_C_COMPILER}
            -DBUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DENABLE_MPI=${MPTENSOR_ENABLE_MPI}
            -DEXPECTED_VERSION=${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
            -DMPIEXEC=${MPIEXEC_EXECUTABLE}
            -DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
            -DPREFLAGS=${MPIEXEC_PREFLAGS}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_package_test.cmake)
  set_tests_properties(package/${mode} PROPERTIES TIMEOUT 1200)
endforeach()
