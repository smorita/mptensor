file(GLOB SOURCES *.cc */*.cc)

add_library(mptensor ${SOURCES})
target_include_directories(mptensor PUBLIC
        $<BUILD_INTERFACE:${mptensor_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
if(ENABLE_MPI)
  target_link_libraries(mptensor PUBLIC ${SCALAPACK_LIBRARIES} MPI::MPI_CXX)
endif()
target_link_libraries(mptensor PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(mptensor PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(mptensor PRIVATE -Wall)

install(TARGETS mptensor
        COMPONENT libraries
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY ${mptensor_SOURCE_DIR}/include/
        DESTINATION include
        PATTERN "index_constructor.py" EXCLUDE
        PATTERN "doxygen_module.hpp" EXCLUDE)
