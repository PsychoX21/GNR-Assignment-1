cmake_minimum_required(VERSION 3.12)
project(DeepNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
endif()

# Find packages
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Check for CUDA (optional)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
    
    set(USE_CUDA ON)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA found - GPU acceleration enabled")
else()
    set(USE_CUDA OFF)
    message(STATUS "CUDA not found - building CPU-only version")
endif()

# Pybind11
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/pybind11/CMakeLists.txt")
    message(FATAL_ERROR "pybind11 not found. Please run: git clone https://github.com/pybind/pybind11.git")
endif()
add_subdirectory(pybind11)

# CUDA architecture detection
if(USE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/deepnet/cpp/include
    ${OpenCV_INCLUDE_DIRS}
)

if(USE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Source files
file(GLOB_RECURSE CPP_SOURCES 
    "deepnet/cpp/src/*.cpp"
    "deepnet/cpp/src/layers/*.cpp"
    "deepnet/cpp/src/optimizers/*.cpp"
)

# CUDA sources (only if CUDA is available)
if(USE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES
        "deepnet/cpp/src/cuda/*.cu"
    )
endif()

# Create C++ library
if(USE_CUDA)
    add_library(deepnet_core STATIC
        ${CPP_SOURCES}
        ${CUDA_SOURCES}
    )
else()
    add_library(deepnet_core STATIC
        ${CPP_SOURCES}
    )
endif()

target_link_libraries(deepnet_core
    OpenMP::OpenMP_CXX
)

if(USE_CUDA)
    target_link_libraries(deepnet_core ${CUDA_LIBRARIES})
    set_target_properties(deepnet_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
else()
    set_target_properties(deepnet_core PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Python bindings
pybind11_add_module(deepnet_backend deepnet/bindings/bindings.cpp)
target_link_libraries(deepnet_backend PRIVATE deepnet_core)

# Installation
install(TARGETS deepnet_backend LIBRARY DESTINATION deepnet)

