@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"
set "PYTHON_EXE=%PROJECT_DIR%\..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: Python environment not found: "%PYTHON_EXE%" 1>&2
  exit /b 2
)
if /I "%~1"=="-NoOpen" (
  call :run %2 %3 %4 %5 %6 %7 %8 %9
) else (
  call :run --open %*
)
exit /b %ERRORLEVEL%

:run
pushd "%PROJECT_DIR%" || exit /b 2
"%PYTHON_EXE%" -m integrations.openpi.rtc_overlap_primary_dashboard %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
