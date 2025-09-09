cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_test)

set(CMAKE_CXX_STANDARD 17)

# Set LibTorch path - using official package
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch")

# Find Torch package
find_package(Torch REQUIRED)

message(STATUS "========================================")
message(STATUS "CUDA SUPPORT ENABLED FOR LIBTORCH")
message(STATUS "Found Torch: ${TORCH_LIBRARIES}")
message(STATUS "========================================")

# Set compiler flags for Windows
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    # Disable some warnings
    add_compile_options(/wd4624 /wd4267 /wd4244 /wd4996 /wd4819)
endif()

# Add executable
add_executable(torch_test main.cpp)

# Link libraries using official method
target_link_libraries(torch_test "${TORCH_LIBRARIES}")

# Set properties
set_property(TARGET torch_test PROPERTY CXX_STANDARD 17)

# Copy DLLs on Windows
if(MSVC)
    file(GLOB TORCH_DLLS "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch/lib/*.dll")
    foreach(DLL ${TORCH_DLLS})
        add_custom_command(TARGET torch_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DLL}
            $<TARGET_FILE_DIR:torch_test>)
    endforeach()
endif()