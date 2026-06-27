@echo off
setlocal

set SCRIPT_DIR=%~dp0
set PS_SCRIPT=%SCRIPT_DIR%Start-Hunyuan3D-Remote-CPU.ps1
if not exist "%PS_SCRIPT%" set PS_SCRIPT=%SCRIPT_DIR%scripts\windows\Start-Hunyuan3D-Remote-CPU.ps1
if not exist "%PS_SCRIPT%" set PS_SCRIPT=%SCRIPT_DIR%Start-Hunyuan3D-Remote.ps1

echo.
echo ==> Starting Hunyuan3D remote CPU launcher
echo ==> PowerShell script: %PS_SCRIPT%
echo ==> Working directory: %CD%
echo.

if not exist "%PS_SCRIPT%" (
  echo ==> Could not find Start-Hunyuan3D-Remote-CPU.ps1
  echo ==> Keep the .cmd and .ps1 files together, or preserve the scripts\windows\ folder structure.
  echo ==> Expected one of:
  echo      %SCRIPT_DIR%Start-Hunyuan3D-Remote-CPU.ps1
  echo      %SCRIPT_DIR%scripts\windows\Start-Hunyuan3D-Remote-CPU.ps1
  echo      %SCRIPT_DIR%Start-Hunyuan3D-Remote.ps1
  echo.
  pause
  exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -RepoRoot "%CD%" -ModelPreset 2.1 -Foreground
set EXIT_CODE=%ERRORLEVEL%

echo.
if not "%EXIT_CODE%"=="0" (
  echo ==> Launcher exited with code %EXIT_CODE%
) else (
  echo ==> Launcher finished successfully
)
echo.
pause
exit /b %EXIT_CODE%
