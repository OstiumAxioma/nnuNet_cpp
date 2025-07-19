@echo off
chcp 65001

echo ========================================
echo nnuNet_cpp Build Script
echo ========================================
echo.

:: 创建构建目录
if not exist build mkdir build
cd build

:: 清理之前的CMake缓存
if exist CMakeCache.txt del CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles

:: 配置项目
echo Configuring project...
cmake .. -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

:: 编译项目
echo.
echo Building project (Release mode)...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

cd ..

echo.
echo ========================================
echo Build completed!
echo ========================================
echo Solution file: build\nnuNet_cpp.sln
echo Executable: build\bin\Release\testToothSegmentation.exe
echo.

:: 复制运行时库
echo Copying runtime libraries from lib\run to build\bin\Release...
if not exist lib\run (
    echo.
    echo ERROR: lib\run directory not found!
    echo This directory must contain all required runtime DLLs.
    echo Please ensure lib\run exists with all necessary ONNX Runtime and CUDA DLLs.
    echo.
    pause
    exit /b 1
)

xcopy /Y /Q lib\run\*.dll build\bin\Release\
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to copy runtime libraries!
    pause
    exit /b 1
)
echo Runtime libraries copied successfully!

:: 确保DentalCbctOnnxSegDLL.dll也被复制
if exist lib\DentalCbctOnnxSegDLL.dll (
    copy /Y lib\DentalCbctOnnxSegDLL.dll build\bin\Release\
    echo DentalCbctOnnxSegDLL.dll copied successfully!
)

echo.
echo All dependencies should now be in place.
echo.
pause