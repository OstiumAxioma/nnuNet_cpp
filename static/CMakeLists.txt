cmake_minimum_required(VERSION 3.20)
project(UnetOnnxSegDLL CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 解决MSVC编译器字符编码警告
if(MSVC)
    add_compile_options("/source-charset:utf-8")
endif()

# 设置ITK目录
set(ITK_DIR "D:/Compile/ITK-5.4.3/lib/cmake/ITK-5.4")

# 查找ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# 设置LibTorch路径
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../lib/libtorch")

# 查找Torch包 - 这会自动设置所有必要的编译标志和依赖
find_package(Torch REQUIRED)

# 显示找到的Torch库
message(STATUS "========================================")
message(STATUS "Found Torch: ${TORCH_LIBRARIES}")
message(STATUS "Torch CXX Flags: ${TORCH_CXX_FLAGS}")
message(STATUS "Torch CUDA: ${TORCH_CUDA_LIBRARIES}")
message(STATUS "Torch cuDNN: ${TORCH_CUDNN_LIBRARY}")
message(STATUS "USE_CUDA: ${USE_CUDA}")
message(STATUS "USE_CUDNN: ${USE_CUDNN}")
message(STATUS "========================================")

# 检查和设置cuDNN
if(TORCH_CUDA_LIBRARIES)
    message(STATUS "LibTorch CUDA libraries found")
    # 设置cuDNN相关变量
    set(CUDNN_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../lib/libtorch/lib/cudnn64_9.dll" CACHE STRING "cuDNN library")
    set(USE_CUDNN ON CACHE BOOL "Use cuDNN")
    message(STATUS "cuDNN library: ${CUDNN_LIBRARY}")
endif()

# 启用CUDA支持（默认启用）
option(USE_CUDA "Enable CUDA support for LibTorch" ON)

# 如果需要CUDA支持
if(USE_CUDA)
    message(STATUS "========================================")
    message(STATUS "CUDA SUPPORT ENABLED FOR LIBTORCH")
    message(STATUS "========================================")
    
    # 添加CUDA编译定义
    add_compile_definitions(USE_CUDA)
    
    # 添加cuDNN相关定义（解决cuDNN is set to 0的问题）
    add_compile_definitions(USE_CUDNN)
    add_compile_definitions(AT_CUDNN_ENABLED)
    
    # 尝试找到CUDA Toolkit以获取版本信息
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "Found CUDA Toolkit: ${CUDA_VERSION}")
        include_directories(${CUDA_INCLUDE_DIRS})
        # 添加CUDA运行时库
        list(APPEND TORCH_LIBRARIES ${CUDA_LIBRARIES})
    else()
        # 如果find_package失败，手动设置CUDA路径
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
        if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/include")
            message(STATUS "Using manual CUDA path: ${CUDA_TOOLKIT_ROOT_DIR}")
            include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
            link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
        endif()
    endif()
else()
    message(WARNING "========================================")
    message(WARNING "CUDA SUPPORT DISABLED FOR LIBTORCH")
    message(WARNING "========================================")
endif()

# 设置C++17标准（LibTorch需要）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")

# 添加LibTorch的编译标志（重要！这包含了所有必要的CUDA相关标志）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 收集源文件
set(SOURCES
    src/UnetSegAI_API.cpp
    src/UnetMain.cpp
    src/UnetPreprocessor.cpp
    src/UnetInference.cpp
    src/UnetTorchInference.cpp
    src/UnetPostprocessor.cpp
    src/UnetIO.cpp
    src/ConfigParser.cpp
    src/dllmain.cpp
    src/pch.cpp
)

set(HEADERS
    header/UnetSegAI_API.h
    header/UnetMain.h
    header/UnetPreprocessor.h
    header/UnetInference.h
    header/UnetTorchInference.h
    header/UnetPostprocessor.h
    header/UnetIO.h
    header/ConfigParser.h
    header/framework.h
    header/pch.h
)

# 创建DLL库
add_library(UnetOnnxSegDLL SHARED ${SOURCES} ${HEADERS})

# 强制每次都重新生成导入库
set_target_properties(UnetOnnxSegDLL PROPERTIES
    LINK_FLAGS "/INCREMENTAL:NO"  # 禁用增量链接
    LINK_DEPENDS_NO_SHARED ON     # 不共享链接依赖
)

# 启用预编译头
target_precompile_headers(UnetOnnxSegDLL PRIVATE header/pch.h)

# 设置包含目录
target_include_directories(UnetOnnxSegDLL PRIVATE
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
target_link_libraries(UnetOnnxSegDLL PRIVATE
    ${ONNXRUNTIME_LIB}
    ${ONNXRUNTIME_PROVIDERS_SHARED_LIB}
    ${ONNXRUNTIME_PROVIDERS_CUDA_LIB}
    ${ITK_LIBRARIES}
    "${TORCH_LIBRARIES}"  # 使用引号，因为TORCH_LIBRARIES可能包含分号分隔的列表
)

# Windows特定设置
if(WIN32)
    # 定义预处理器宏
    target_compile_definitions(UnetOnnxSegDLL PRIVATE
        UNETONNXSEGDLL_EXPORTS
        _WINDOWS
        _USRDLL
        _UNICODE
        UNICODE
        NOMINMAX  # 防止Windows.h定义min/max宏
    )
    
    # 设置编译选项
    target_compile_options(UnetOnnxSegDLL PRIVATE
        /W3     # 警告级别3
        /EHsc   # 启用C++异常处理
        /MP     # 多处理器编译
        /wd4251 # 禁用C4251警告（LibTorch的STL导出警告）
        /wd4275 # 禁用C4275警告（非DLL接口类用作基类）
        /wd4244 # 禁用C4244警告（类型转换警告）
        /wd4267 # 禁用C4267警告（size_t到int转换）
    )
    
    # Release配置优化
    target_compile_options(UnetOnnxSegDLL PRIVATE
        $<$<CONFIG:Release>:/O2>    # 最大优化
        $<$<CONFIG:Release>:/GL>    # 全程序优化
    )
    
    # Debug配置
    target_compile_options(UnetOnnxSegDLL PRIVATE
        $<$<CONFIG:Debug>:/Od>      # 禁用优化
        $<$<CONFIG:Debug>:/Zi>      # 生成调试信息
    )
    
    # 链接Windows库
    target_link_libraries(UnetOnnxSegDLL PRIVATE
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
add_custom_command(TARGET UnetOnnxSegDLL POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:UnetOnnxSegDLL>
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_LINKER_FILE:UnetOnnxSegDLL>
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/
)

# 显示配置信息
message(STATUS "Building UnetOnnxSegDLL")
message(STATUS "Output directory: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")