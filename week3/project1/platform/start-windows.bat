@echo off
rem Double-click to start the tile inspection platform on Windows.
rem Runs run.ps1 without changing the PowerShell execution policy of the machine.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 (
  echo.
  echo The platform stopped with an error - see the messages above.
  pause
)
