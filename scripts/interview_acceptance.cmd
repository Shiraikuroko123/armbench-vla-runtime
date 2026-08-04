@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0interview_acceptance.ps1" %*
exit /b %ERRORLEVEL%
