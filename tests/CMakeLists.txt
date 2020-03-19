file(GLOB SOURCES *.cc)
add_executable(tensor_test ${SOURCES})
target_link_libraries(tensor_test mptensor)

add_subdirectory(file_io)
