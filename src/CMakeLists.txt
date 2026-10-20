file(GLOB SOURCES *.cc */*.cc)

add_library(mptensor ${SOURCES})
add_library(mptensor::mptensor ALIAS mptensor)
target_include_directories(mptensor PUBLIC
        $<BUILD_INTERFACE:${mptensor_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(mptensor PUBLIC cxx_std_17)
if(MPTENSOR_ENABLE_MPI)
  target_link_libraries(mptensor PUBLIC ${SCALAPACK_LIBRARIES} MPI::MPI_CXX)
else()
  # Users of a non-MPI build need _NO_MPI as well, so it is a usage requirement.
  target_compile_definitions(mptensor PUBLIC _NO_MPI)
endif()
target_link_libraries(mptensor PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(mptensor PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(mptensor PRIVATE -Wall)

if(MPTENSOR_INSTALL)
  include(CMakePackageConfigHelpers)
  set(MPTENSOR_CMAKE_DIR lib/cmake/mptensor)

  install(TARGETS mptensor
          EXPORT mptensorTargets
          COMPONENT libraries
          ARCHIVE DESTINATION lib
          LIBRARY DESTINATION lib)
  install(DIRECTORY ${mptensor_SOURCE_DIR}/include/
          DESTINATION include)

  install(EXPORT mptensorTargets
          NAMESPACE mptensor::
          DESTINATION ${MPTENSOR_CMAKE_DIR})
  configure_package_config_file(
          ${mptensor_SOURCE_DIR}/cmake/mptensorConfig.cmake.in
          ${CMAKE_CURRENT_BINARY_DIR}/mptensorConfig.cmake
          INSTALL_DESTINATION ${MPTENSOR_CMAKE_DIR})
  # 0.x releases are not compatible across minor versions.
  write_basic_package_version_file(
          ${CMAKE_CURRENT_BINARY_DIR}/mptensorConfigVersion.cmake
          VERSION ${PROJECT_VERSION}
          COMPATIBILITY SameMinorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mptensorConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/mptensorConfigVersion.cmake
          DESTINATION ${MPTENSOR_CMAKE_DIR})
endif()
