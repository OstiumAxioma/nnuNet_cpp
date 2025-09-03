@echo off
chcp 65001

echo ========================================
echo UnetOnnxSegDLL Build Script
echo ========================================
echo.

:: 创建构建目录
if not exist build mkdir build
cd build

:: 完全清理之前的构建（强制重新生成所有文件）
echo Cleaning previous build completely...
if exist CMakeCache.txt del CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles
if exist *.vcxproj del /q *.vcxproj
if exist *.vcxproj.filters del /q *.vcxproj.filters
if exist *.sln del /q *.sln
if exist x64 rmdir /s /q x64
if exist Release rmdir /s /q Release
if exist Debug rmdir /s /q Debug
if exist bin rmdir /s /q bin
if exist lib rmdir /s /q lib

:: 清理Visual Studio和CMake的中间文件
if exist UnetOnnxSegDLL.dir rmdir /s /q UnetOnnxSegDLL.dir
if exist *.exp del /q *.exp
if exist *.ilk del /q *.ilk
if exist *.pdb del /q *.pdb
if exist .vs rmdir /s /q .vs

:: 配置项目
echo Configuring UnetOnnxSegDLL project...
cmake .. -G "Visual Studio 17 2022" -A x64 -DITK_DIR=D:/Compile/ITK-5.4.3/lib/cmake/ITK-5.4

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

:: 编译项目（只编译Release版本，因为ITK是Release版本）
echo.
echo Building DLL (Release mode)...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Release build failed!
    pause
    exit /b 1
)

:: 注释掉Debug编译，因为ITK库是Release版本
:: echo.
:: echo Building DLL (Debug mode)...
:: cmake --build . --config Debug
:: 
:: if %ERRORLEVEL% NEQ 0 (
::     echo Debug build failed!
::     pause
::     exit /b 1
:: )

cd ..

echo.
echo ========================================
echo Build completed!
echo ========================================
echo.

:: 复制编译生成的文件到主项目lib目录
echo Copying generated files to main project lib directory...
if exist "build\bin\Release\UnetOnnxSegDLL.dll" (
    copy /Y "build\bin\Release\UnetOnnxSegDLL.dll" "..\lib\"
    echo UnetOnnxSegDLL.dll copied to ..\lib\
)
if exist "build\lib\Release\UnetOnnxSegDLL.lib" (
    copy /Y "build\lib\Release\UnetOnnxSegDLL.lib" "..\lib\"
    echo UnetOnnxSegDLL.lib copied to ..\lib\
)

:: 确保目标目录存在
if not exist "..\build\bin\Release" mkdir "..\build\bin\Release"
if not exist "..\build\bin\Debug" mkdir "..\build\bin\Debug"

:: 从lib目录拷贝文件到目标目录
echo Copying files from lib directory...

:: 拷贝Release版本
if exist "..\lib\UnetOnnxSegDLL.dll" (
    copy "..\lib\UnetOnnxSegDLL.dll" "..\build\bin\Release\"
    echo Release DLL copied to ..\build\bin\Release\
) else (
    echo Warning: ..\lib\UnetOnnxSegDLL.dll not found!
)

if exist "..\lib\UnetOnnxSegDLL.lib" (
    copy "..\lib\UnetOnnxSegDLL.lib" "..\build\bin\Release\"
    echo Release LIB copied to ..\build\bin\Release\
) else (
    echo Warning: ..\lib\UnetOnnxSegDLL.lib not found!
)

:: 拷贝Debug版本（如果存在）
if exist "..\lib\UnetOnnxSegDLL_d.dll" (
    copy "..\lib\UnetOnnxSegDLL_d.dll" "..\build\bin\Debug\"
    echo Debug DLL copied to ..\build\bin\Debug\
)

if exist "..\lib\UnetOnnxSegDLL_d.lib" (
    copy "..\lib\UnetOnnxSegDLL_d.lib" "..\build\bin\Debug\"
    echo Debug LIB copied to ..\build\bin\Debug\
)

echo.
echo All files copied successfully!
echo Target directory: ..\build\bin\Release\
echo.
pause