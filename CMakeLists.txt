cmake_minimum_required(VERSION 3.20)
project(nnuNet_cpp CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置ITK目录
set(ITK_DIR "D:/Compile/ITK-5.4.3/lib/cmake/ITK-5.4")

# 查找ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 添加可执行文件
add_executable(testToothSegmentation src/testToothSegmentation.cpp)

# 设置包含目录
target_include_directories(testToothSegmentation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/header
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/CImg
)

# 查找并链接库文件
# ONNX Runtime库
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime/lib NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_PROVIDERS_SHARED_LIB onnxruntime_providers_shared PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime/lib NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_PROVIDERS_CUDA_LIB onnxruntime_providers_cuda PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime/lib NO_DEFAULT_PATH)

# UnetOnnxSegDLL库
find_library(UNET_SEG_LIB UnetOnnxSegDLL PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH)

# 链接库文件
target_link_libraries(testToothSegmentation PRIVATE
    ${UNET_SEG_LIB}
    ${ONNXRUNTIME_LIB}
    ${ONNXRUNTIME_PROVIDERS_SHARED_LIB}
    ${ONNXRUNTIME_PROVIDERS_CUDA_LIB}
    ${ITK_LIBRARIES}
)

# Windows特定设置
if(WIN32)
    # 设置Windows子系统
    set_property(TARGET testToothSegmentation PROPERTY WIN32_EXECUTABLE FALSE)
    
    # 定义预处理器宏
    target_compile_definitions(testToothSegmentation PRIVATE
        _WINDOWS
        _UNICODE
        UNICODE
    )
    
    # 添加Windows库
    target_link_libraries(testToothSegmentation PRIVATE
        kernel32
        user32
        gdi32
        winspool
        comdlg32
        advapi32
        shell32
        ole32
        oleaut32
        uuid
        odbc32
        odbccp32
    )
    
    # 设置编译选项
    target_compile_options(testToothSegmentation PRIVATE
        /W3     # 警告级别3
        /EHsc   # 启用C++异常处理
        /MP     # 多处理器编译
    )
    
    # Release配置优化
    target_compile_options(testToothSegmentation PRIVATE
        $<$<CONFIG:Release>:/O2>    # 最大优化
        $<$<CONFIG:Release>:/GL>    # 全程序优化
    )
    
    # Debug配置
    target_compile_options(testToothSegmentation PRIVATE
        $<$<CONFIG:Debug>:/Od>      # 禁用优化
        $<$<CONFIG:Debug>:/Zi>      # 生成调试信息
    )
endif()

# 复制DLL文件到输出目录（运行时需要）
if(WIN32)
    # 复制ONNX Runtime DLL文件 - 必须从lib/run目录复制
    # 检查lib/run目录是否存在
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/run)
        message(FATAL_ERROR "Required directory lib/run not found! This directory should contain all runtime DLLs.")
    endif()
    
    # 从lib/run复制所有DLL文件
    file(GLOB RUNTIME_DLLS ${CMAKE_CURRENT_SOURCE_DIR}/lib/run/*.dll)
    if(NOT RUNTIME_DLLS)
        message(FATAL_ERROR "No DLL files found in lib/run directory!")
    endif()
    
    foreach(dll_file ${RUNTIME_DLLS})
        add_custom_command(TARGET testToothSegmentation POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${dll_file}
                $<TARGET_FILE_DIR:testToothSegmentation>
        )
    endforeach()
    message(STATUS "Found ${CMAKE_CURRENT_SOURCE_DIR}/lib/run directory with runtime DLLs")
    
    # 复制UnetOnnxSegDLL.dll
    add_custom_command(TARGET testToothSegmentation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/lib/UnetOnnxSegDLL.dll
            $<TARGET_FILE_DIR:testToothSegmentation>
    )
endif()

# 设置VS工程的启动项目
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT testToothSegmentation)

# 显示配置信息
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")