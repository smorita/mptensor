file(GLOB SOURCES *.cc */*.cc)

add_library(mptensor ${SOURCES})
target_include_directories(mptensor PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(mptensor ${SCALAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_compile_options(mptensor PRIVATE -Wall)

install(TARGETS mptensor
        COMPONENT libraries
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION include
        PATTERN "index_constructor.py" EXCLUDE
        PATTERN "doxygen_module.hpp" EXCLUDE)
