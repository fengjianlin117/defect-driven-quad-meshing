cmake_minimum_required(VERSION 3.20)

project(trimesh2quad_backend_rebuild LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Repository root")
set(MIQ_BACKEND_ROOT "" CACHE PATH "Root containing libs/libigl, libs/CoMISo, and libs/libQEx")
set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Directory containing Eigen/Core")
set(OPENMESH_INCLUDE_DIR "" CACHE PATH "Directory containing OpenMesh/Core")
set(OPENMESH_LIBRARY_DIR "" CACHE PATH "Directory containing OpenMesh libraries")
set(LIBQEX_STATIC_LIBRARY "" CACHE FILEPATH "Path to libQExStatic.a")

if(NOT MIQ_BACKEND_ROOT)
    message(FATAL_ERROR "MIQ_BACKEND_ROOT is required. Run scripts/build.sh or pass -DMIQ_BACKEND_ROOT=...")
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(NOT EIGEN3_INCLUDE_DIR)
    find_path(_EIGEN3_INCLUDE_DIR
        NAMES Eigen/Core
        HINTS /usr/include/eigen3 /usr/local/include/eigen3
        REQUIRED
    )
    set(EIGEN3_INCLUDE_DIR "${_EIGEN3_INCLUDE_DIR}" CACHE PATH "Directory containing Eigen/Core" FORCE)
endif()

if(NOT OPENMESH_INCLUDE_DIR)
    find_path(_OPENMESH_INCLUDE_DIR
        NAMES OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh
        HINTS /usr/include /usr/local/include
        REQUIRED
    )
    set(OPENMESH_INCLUDE_DIR "${_OPENMESH_INCLUDE_DIR}" CACHE PATH "Directory containing OpenMesh/Core" FORCE)
endif()

find_library(OPENMESH_CORE_LIBRARY
    NAMES OpenMeshCore
    HINTS "${OPENMESH_LIBRARY_DIR}" /usr/lib/x86_64-linux-gnu /usr/local/lib
    REQUIRED
)

find_library(OPENMESH_TOOLS_LIBRARY
    NAMES OpenMeshTools
    HINTS "${OPENMESH_LIBRARY_DIR}" /usr/lib/x86_64-linux-gnu /usr/local/lib
    REQUIRED
)

find_path(LIBIGL_INCLUDE_DIR
    NAMES igl/readOBJ.h
    HINTS "${MIQ_BACKEND_ROOT}/libs/libigl/include"
    NO_DEFAULT_PATH
    REQUIRED
)

find_path(COMISO_PARENT_INCLUDE_DIR
    NAMES CoMISo/Solver/MISolver.hh
    HINTS "${MIQ_BACKEND_ROOT}/libs"
    NO_DEFAULT_PATH
    REQUIRED
)

find_path(GMM_INCLUDE_DIR
    NAMES gmm/gmm.h
    HINTS "${MIQ_BACKEND_ROOT}/libs/CoMISo/ext/gmm-4.2/include"
    NO_DEFAULT_PATH
    REQUIRED
)

find_path(LIBQEX_INCLUDE_DIR
    NAMES qex.h
    HINTS "${MIQ_BACKEND_ROOT}/libs/libQEx/interfaces/c"
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT LIBQEX_STATIC_LIBRARY)
    find_library(_LIBQEX_STATIC_LIBRARY
        NAMES QExStatic
        HINTS "${MIQ_BACKEND_ROOT}/build/libqex"
        NO_DEFAULT_PATH
        REQUIRED
    )
    set(LIBQEX_STATIC_LIBRARY "${_LIBQEX_STATIC_LIBRARY}" CACHE FILEPATH "Path to libQExStatic.a" FORCE)
endif()

set(MIQ_WRAPPER_SOURCE
    "${REPO_ROOT}/backend/src/backend/full_miq/run_miq_from_field_full.cpp")
set(QEX_WRAPPER_SOURCE
    "${REPO_ROOT}/backend/src/backend/full_miq/run_qex_from_miq.cpp")
set(COMISO_SOURCES
    "${MIQ_BACKEND_ROOT}/libs/CoMISo/Solver/MISolver.cc"
    "${MIQ_BACKEND_ROOT}/libs/CoMISo/Solver/ConstrainedSolver.cc"
    "${MIQ_BACKEND_ROOT}/libs/CoMISo/Solver/GMM_Tools.cc"
    "${MIQ_BACKEND_ROOT}/libs/CoMISo/Solver/IterativeSolverT.cc"
    "${MIQ_BACKEND_ROOT}/libs/CoMISo/Solver/EigenLDLTSolver.cc"
)

foreach(required_file
        "${MIQ_WRAPPER_SOURCE}"
        "${QEX_WRAPPER_SOURCE}"
        "${LIBQEX_STATIC_LIBRARY}")
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Required backend file is missing: ${required_file}")
    endif()
endforeach()

foreach(comiso_source IN LISTS COMISO_SOURCES)
    if(NOT EXISTS "${comiso_source}")
        message(FATAL_ERROR "Required CoMISo source is missing: ${comiso_source}")
    endif()
endforeach()

add_executable(miq_adapter "${MIQ_WRAPPER_SOURCE}" ${COMISO_SOURCES})
target_include_directories(miq_adapter PRIVATE
    "${EIGEN3_INCLUDE_DIR}"
    "${LIBIGL_INCLUDE_DIR}"
    "${COMISO_PARENT_INCLUDE_DIR}"
    "${GMM_INCLUDE_DIR}"
)
target_compile_definitions(miq_adapter PRIVATE INCLUDE_TEMPLATES)
target_compile_options(miq_adapter PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-deprecated-declarations;-Wno-register>
)
target_link_libraries(miq_adapter PRIVATE
    BLAS::BLAS
    LAPACK::LAPACK
    OpenMP::OpenMP_CXX
)

add_executable(qex_adapter "${QEX_WRAPPER_SOURCE}")
target_include_directories(qex_adapter PRIVATE
    "${EIGEN3_INCLUDE_DIR}"
    "${LIBIGL_INCLUDE_DIR}"
    "${LIBQEX_INCLUDE_DIR}"
    "${OPENMESH_INCLUDE_DIR}"
)
target_link_libraries(qex_adapter PRIVATE
    "${LIBQEX_STATIC_LIBRARY}"
    "${OPENMESH_CORE_LIBRARY}"
    "${OPENMESH_TOOLS_LIBRARY}"
    OpenMP::OpenMP_CXX
)

set_target_properties(miq_adapter qex_adapter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "Dependency root: ${MIQ_BACKEND_ROOT}")
message(STATUS "libigl include: ${LIBIGL_INCLUDE_DIR}")
message(STATUS "CoMISo include: ${COMISO_PARENT_INCLUDE_DIR}")
message(STATUS "libQEx static: ${LIBQEX_STATIC_LIBRARY}")
message(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "OpenMesh: ${OPENMESH_INCLUDE_DIR}")

enable_testing()
add_test(NAME miq_adapter_usage COMMAND miq_adapter)
set_tests_properties(miq_adapter_usage PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME qex_adapter_usage COMMAND qex_adapter)
set_tests_properties(qex_adapter_usage PROPERTIES
    WILL_FAIL TRUE
)

add_executable(miq_adaptive "${CMAKE_CURRENT_SOURCE_DIR}/src/adaptive/run_weak_layout_miq.cpp" ${COMISO_SOURCES})
target_include_directories(miq_adaptive PRIVATE "${EIGEN3_INCLUDE_DIR}" "${LIBIGL_INCLUDE_DIR}" "${COMISO_PARENT_INCLUDE_DIR}" "${GMM_INCLUDE_DIR}")
target_compile_definitions(miq_adaptive PRIVATE INCLUDE_TEMPLATES)
target_compile_options(miq_adaptive PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-deprecated-declarations;-Wno-register>)
target_link_libraries(miq_adaptive PRIVATE BLAS::BLAS LAPACK::LAPACK OpenMP::OpenMP_CXX)
set_target_properties(miq_adaptive PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME adaptive_capabilities COMMAND miq_adaptive --capabilities)
