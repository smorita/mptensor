file(GLOB SOURCES mptensor/*.cc mptensor/lapack/*.cc mptensor/matrix_interface/*.cc mptensor/scalapack/*.cc)
file(GLOB HEADERS *.hpp mptensor/*.hpp mptensor/lapack/*.hpp mptensor/matrix_interface/*.hpp mptensor/scalapack/*.hpp)

add_library(mptensor ${SOURCES})
target_link_libraries(mptensor ${SCALAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${MPI_CXX_LIBRARIES})

install(TARGETS mptensor COMPONENT libraries ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY . COMPONENT headers DESTINATION include FILES_MATCHING PATTERN "*.hpp")
