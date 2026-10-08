add_executable(test_index basic_index.cc conversions.cc)
target_link_libraries(test_index PRIVATE mptensor_gtest_main)

mptensor_add_test(index/test_index test_index)
