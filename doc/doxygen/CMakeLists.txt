set(MPTENSOR_DOC_INSTALL_DIR share/mptensor/v${mptensor_VERSION})
set(MPTENSOR_PDF_FILE ${mptensor_BINARY_DIR}/mptensor.pdf)

find_package(Doxygen REQUIRED)
configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

add_custom_target(doxygen ALL COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${MPTENSOR_DOC_INSTALL_DIR})

option(BUILD_PDF_DOCS "Build PDF documentation (requires LaTeX)" OFF)
if(BUILD_PDF_DOCS)
  find_package(LATEX REQUIRED)
  add_custom_target(doxygen_pdf ALL
    COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}/latex
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf ${MPTENSOR_PDF_FILE}
    DEPENDS doxygen
  )
  install(FILES ${MPTENSOR_PDF_FILE} DESTINATION ${MPTENSOR_DOC_INSTALL_DIR})
else()
  message(STATUS "PDF documentation is disabled. To enable, re-run cmake with -DBUILD_DOCS=ON -DBUILD_PDF_DOCS=ON.")
endif()
