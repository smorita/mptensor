file(GLOB SOURCES *.cc)
set(testname test_all)
add_executable(${testname} ${SOURCES})
target_link_libraries(${testname} mptensor)

if(ENABLE_MPI)
  add_test(NAME ${testname}_mpi0001
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:${testname}>)
  add_test(NAME ${testname}_mpi0004
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:${testname}>)
else()
  add_test(NAME ${testname}_serial COMMAND $<TARGET_FILE:${testname}>)
endif()

add_subdirectory(file_io)
