cmake_minimum_required(VERSION 3.20)
project(DentalCbctOnnxSegDLL CXX)

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

# 收集源文件
set(SOURCES
    src/DentalCbctSegAI_API.cpp
    src/DentalUnet.cpp
    src/ConfigParser.cpp
    src/dllmain.cpp
    src/pch.cpp
)

set(HEADERS
    header/DentalCbctSegAI_API.h
    header/DentalUnet.h
    header/ConfigParser.h
    header/framework.h
    header/pch.h
)

# 创建DLL库
add_library(DentalCbctOnnxSegDLL SHARED ${SOURCES} ${HEADERS})

# 启用预编译头
target_precompile_headers(DentalCbctOnnxSegDLL PRIVATE header/pch.h)

# 设置包含目录
target_include_directories(DentalCbctOnnxSegDLL PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/header
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/onnxruntime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/CImg
)

# 查找并链接库文件
# ONNX Runtime库
find_library(ONNXRUNTIME_LIB onnxruntime 
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../lib/onnxruntime/lib 
    NO_DEFAULT_PATH
)
find_library(ONNXRUNTIME_PROVIDERS_SHARED_LIB onnxruntime_providers_shared 
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../lib/onnxruntime/lib 
    NO_DEFAULT_PATH
)
find_library(ONNXRUNTIME_PROVIDERS_CUDA_LIB onnxruntime_providers_cuda 
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../lib/onnxruntime/lib 
    NO_DEFAULT_PATH
)

# 链接库文件
target_link_libraries(DentalCbctOnnxSegDLL PRIVATE
    ${ONNXRUNTIME_LIB}
    ${ONNXRUNTIME_PROVIDERS_SHARED_LIB}
    ${ONNXRUNTIME_PROVIDERS_CUDA_LIB}
    ${ITK_LIBRARIES}
)

# Windows特定设置
if(WIN32)
    # 定义预处理器宏
    target_compile_definitions(DentalCbctOnnxSegDLL PRIVATE
        DENTALCBCTSEGDLL_EXPORTS
        _WINDOWS
        _USRDLL
        _UNICODE
        UNICODE
        NOMINMAX  # 防止Windows.h定义min/max宏
    )
    
    # 设置编译选项
    target_compile_options(DentalCbctOnnxSegDLL PRIVATE
        /W3     # 警告级别3
        /EHsc   # 启用C++异常处理
        /MP     # 多处理器编译
    )
    
    # Release配置优化
    target_compile_options(DentalCbctOnnxSegDLL PRIVATE
        $<$<CONFIG:Release>:/O2>    # 最大优化
        $<$<CONFIG:Release>:/GL>    # 全程序优化
    )
    
    # Debug配置
    target_compile_options(DentalCbctOnnxSegDLL PRIVATE
        $<$<CONFIG:Debug>:/Od>      # 禁用优化
        $<$<CONFIG:Debug>:/Zi>      # 生成调试信息
    )
    
    # 链接Windows库
    target_link_libraries(DentalCbctOnnxSegDLL PRIVATE
        kernel32
        user32
        gdi32
        advapi32
        shell32
        ole32
        oleaut32
        uuid
    )
endif()

# 复制DLL到主项目的lib目录（构建后）
add_custom_command(TARGET DentalCbctOnnxSegDLL POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:DentalCbctOnnxSegDLL>
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_LINKER_FILE:DentalCbctOnnxSegDLL>
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/
)

# 显示配置信息
message(STATUS "Building DentalCbctOnnxSegDLL")
message(STATUS "Output directory: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")