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
echo Note: If you get DentalCbctOnnxSegDLL.dll missing error,
echo       please copy it manually to build\bin\Release\
echo.
pause