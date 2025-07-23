@echo off
echo Converting all raw files to numpy format...
echo.

REM 转换所有结果目录中的raw文件
python convert_raw_to_npy.py ..\result -r -v

echo.
echo Conversion complete!
echo.
echo You can now visualize the results using:
echo   python visualize_results.py ..\result
echo.
pause