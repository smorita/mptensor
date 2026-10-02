set(FILE_IO_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/data)
file(MAKE_DIRECTORY ${FILE_IO_DATA_DIR})

add_executable(test_save save.cc)
target_link_libraries(test_save PRIVATE mptensor_gtest_main)
add_executable(test_load load.cc)
target_link_libraries(test_load PRIVATE mptensor_gtest_main)

# prepare -> save (all process counts) -> load (all process counts)
add_test(NAME file_io/prepare
  COMMAND ${CMAKE_COMMAND} -DDATA_DIR=${FILE_IO_DATA_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/prepare_data_dir.cmake)
set_tests_properties(file_io/prepare PROPERTIES FIXTURES_SETUP file_io_dir)

mptensor_add_test(file_io/save test_save
  WORKING_DIRECTORY ${FILE_IO_DATA_DIR}
  FIXTURES_REQUIRED file_io_dir
  FIXTURES_SETUP file_io_data)
mptensor_add_test(file_io/load test_load
  WORKING_DIRECTORY ${FILE_IO_DATA_DIR}
  FIXTURES_REQUIRED file_io_data)
