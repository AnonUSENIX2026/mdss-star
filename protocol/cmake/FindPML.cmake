if(PML_DIR)
  find_path(
    PML_LIB
    NAMES mat_lzz_pX_forms.o
    PATHS ${PML_DIR}
    PATH_SUFFIXES lib/PML
    NO_DEFAULT_PATH
    DOC "PML library"
  )

  find_path(
    PML_HEADERS
    NAMES mat_lzz_pX_forms.h
    PATHS ${PML_DIR}
    PATH_SUFFIXES include/PML
    NO_DEFAULT_PATH
    DOC "PML headers"
  )
else(PML_DIR)
  find_path(
    PML_LIB
    NAMES mat_lzz_pX_forms.o
    PATH_SUFFIXES lib/PML
    DOC "PML library"
  )

  find_path(
    PML_HEADERS
    NAMES mat_lzz_pX_forms.h
    PATH_SUFFIXES include/PML
    DOC "PML headers"
  )
endif(PML_DIR)

if (PML_HEADERS AND PML_LIB)
  set(PML_FOUND TRUE)
  file(GLOB PML_OBJ_FILES ${PML_LIB}/*.o)
else()
  set(PML_FOUND FALSE)
endif()

if (PML_FOUND)
    add_library(PML INTERFACE IMPORTED)

    target_include_directories(PML INTERFACE ${PML_HEADERS}/..)
    target_link_libraries(PML INTERFACE ${PML_OBJ_FILES})

    message(STATUS "Found PML: Headers at ${PML_HEADERS}, Objects at ${PML_LIB}")
else()
    message(FATAL_ERROR "Could NOT find PML.")
endif()
