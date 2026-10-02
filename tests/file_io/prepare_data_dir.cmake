# Usage: cmake -DDATA_DIR=<dir> -P prepare_data_dir.cmake
# Starts every file_io run from an empty directory, so that stale files
# from an earlier run cannot hide a failure of save().
if(NOT DATA_DIR)
  message(FATAL_ERROR "DATA_DIR is not set")
endif()
file(REMOVE_RECURSE "${DATA_DIR}")
file(MAKE_DIRECTORY "${DATA_DIR}")
