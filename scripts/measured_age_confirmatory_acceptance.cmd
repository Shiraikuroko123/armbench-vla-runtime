@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0measured_age_confirmatory_acceptance.ps1" %*
exit /b %ERRORLEVEL%
