cmake_minimum_required(VERSION 3.16...3.31)
project(mptensor_package_consumer LANGUAGES CXX)

# Either use an installed mptensor (find_package) or build it as a
# subproject (add_subdirectory), as a project with a git submodule would.
if(MPTENSOR_SOURCE_DIR)
  add_subdirectory(${MPTENSOR_SOURCE_DIR} mptensor)
else()
  find_package(mptensor ${EXPECTED_VERSION} REQUIRED)
  message(STATUS "mptensor ${mptensor_VERSION}, MPI: ${mptensor_ENABLE_MPI}")
endif()

add_executable(consumer main.cc)
target_link_libraries(consumer PRIVATE mptensor::mptensor)

install(TARGETS consumer DESTINATION bin)
