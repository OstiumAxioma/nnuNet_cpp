@echo off
chcp 65001

echo ========================================
echo DentalCbctOnnxSegDLL Build Script
echo ========================================
echo.

:: 创建构建目录
if not exist build mkdir build
cd build

:: 清理之前的CMake缓存
if exist CMakeCache.txt del CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles

:: 配置项目
echo Configuring DLL project...
cmake .. -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

:: 编译项目（Release和Debug两个版本）
echo.
echo Building DLL (Release mode)...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Release build failed!
    pause
    exit /b 1
)

echo.
echo Building DLL (Debug mode)...
cmake --build . --config Debug

if %ERRORLEVEL% NEQ 0 (
    echo Debug build failed!
    pause
    exit /b 1
)

cd ..

echo.
echo ========================================
echo Build completed!
echo ========================================
echo.

:: 确保目标目录存在
if not exist "..\build\bin\Release" mkdir "..\build\bin\Release"
if not exist "..\build\bin\Debug" mkdir "..\build\bin\Debug"

:: 从lib目录拷贝文件到目标目录
echo Copying files from lib directory...

:: 拷贝Release版本
if exist "..\lib\DentalCbctOnnxSegDLL.dll" (
    copy "..\lib\DentalCbctOnnxSegDLL.dll" "..\build\bin\Release\"
    echo Release DLL copied to ..\build\bin\Release\
) else (
    echo Warning: ..\lib\DentalCbctOnnxSegDLL.dll not found!
)

if exist "..\lib\DentalCbctOnnxSegDLL.lib" (
    copy "..\lib\DentalCbctOnnxSegDLL.lib" "..\build\bin\Release\"
    echo Release LIB copied to ..\build\bin\Release\
) else (
    echo Warning: ..\lib\DentalCbctOnnxSegDLL.lib not found!
)

:: 拷贝Debug版本（如果存在）
if exist "..\lib\DentalCbctOnnxSegDLL_d.dll" (
    copy "..\lib\DentalCbctOnnxSegDLL_d.dll" "..\build\bin\Debug\"
    echo Debug DLL copied to ..\build\bin\Debug\
)

if exist "..\lib\DentalCbctOnnxSegDLL_d.lib" (
    copy "..\lib\DentalCbctOnnxSegDLL_d.lib" "..\build\bin\Debug\"
    echo Debug LIB copied to ..\build\bin\Debug\
)

echo.
echo All files copied successfully!
echo Target directory: ..\build\bin\Release\
echo.
pause