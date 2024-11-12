# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name
project(imageStitching)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#change to your path
set(CMAKE_TOOLCHAIN_FILE "/opt/homebrew/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")


# Find OpenCV package

find_package(OpenCV REQUIRED)

# Add executable target
add_executable(imageStitching image_Stitching.cpp)

# Link OpenCV libraries
target_link_libraries(imageStitching ${OpenCV_LIBS})

# Link the filesystem library if using GNU compiler
if(CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(imageStitching stdc++fs)
endif()
