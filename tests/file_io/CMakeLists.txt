set(PROGS save load)
foreach(name ${PROGS})
  add_executable(test_${name} ${name}.cc ../mpi_tool.cc)
  target_link_libraries(test_${name} mptensor)
endforeach(name ${PROGS})

if(ENABLE_MPI)
  add_test(NAME test_save_mpi0001
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:test_save>)
  add_test(NAME test_save_mpi0002
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_save>)
  add_test(NAME test_save_mpi0003
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 $<TARGET_FILE:test_save>)
  add_test(NAME test_save_mpi0004
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:test_save>)
  add_test(NAME test_load_mpi0001
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:test_load>)
  add_test(NAME test_load_mpi0002
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_load>)
  add_test(NAME test_load_mpi0003
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 $<TARGET_FILE:test_load>)
  add_test(NAME test_load_mpi0004
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:test_load>)
else()
  add_test(NAME test_save_serial COMMAND test_save)
  add_test(NAME test_load_serial COMMAND test_load)
endif()
