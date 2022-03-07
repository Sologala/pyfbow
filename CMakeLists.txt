cmake_minimum_required(VERSION 3.4...3.18)
project(pyfbow)



IF(NOT UNIX)
  SET(OpenCV_USE_STATIC_LIBS ON)
ENDIF()
find_package(OpenCV REQUIRED)


SET(fbow_USE_STATIC_LIBS ON)
find_package(fbow REQUIRED
          PATHS install_local
          NO_DEFAULT_PATH)
message("fbow Found  " ${fbow_FOUND})



# Building options
#-------------------------------------------------------------------------------
option(BUILD_PYTHON3 "Build fbow Python3 binding instead of default Python2." ON)
OPTION(BUILD_STATICALLY_LINKED  "Set to OFF to link with opencv and fbow dynamically" OFF)


if(BUILD_PYTHON3)
    set(PYTHON_VERSION 3)
else()
    set(PYTHON_VERSION 2)
endif()

message( STATUS "BUILD_PYTHON3 = ${BUILD_PYTHON3}")
message( STATUS "PYTHON_VERSION = ${PYTHON_VERSION}")




#-----------------------------------------------------------------------------------------------------------------------
# Python binding
#-----------------------------------------------------------------------------------------------------------------------

# FIND_PACKAGE(PythonInterp 3 REQUIRED)
# FIND_PACKAGE(PythonLibs 3 REQUIRED)

# Add cmake_modules to module
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NumPy REQUIRED)
# message(${fbow_INCLUDE_DIRS} ${fbow_LIBS})

if (UNIX)
  INCLUDE_DIRECTORIES(
  ${OpenCV_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDES}
  ${fbow_INCLUDE_DIRS}/fbow
 )
else()
  INCLUDE_DIRECTORIES(
  ${OpenCV_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDES}
  ${fbow_INCLUDE_DIRS}/fbow
 )
endif()



add_subdirectory(thirdparty/pybind11)

include_directories("thirdparty/pybind11/include")
include_directories("thirdparty/pybind11_opencv_numpy")

pybind11_add_module(pyfbow "src/fbow_wrapper.cpp" "./thirdparty/pybind11_opencv_numpy/ndarray_converter.cpp")
target_link_libraries(pyfbow PRIVATE 
    ${OpenCV_LIBRARIES}    ${fbow_LIBS})
