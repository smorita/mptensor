add_executable(test_tensor transpose.cc)
target_sources(test_tensor PRIVATE reshape.cc slice.cc set_slice.cc)
target_link_libraries(test_tensor PRIVATE mptensor_gtest_main)

mptensor_add_test(tensor/test_tensor test_tensor)
