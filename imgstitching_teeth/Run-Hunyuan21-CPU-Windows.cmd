@echo off
setlocal
chcp 65001 >nul

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"
set "PS_SCRIPT=%REPO_ROOT%\scripts\windows\Run-Hunyuan21-CPU-Windows.ps1"

echo.
echo ==> Hunyuan3D-2.1 CPU one-click launcher
echo ==> Repo root: %REPO_ROOT%
echo ==> PowerShell script: %PS_SCRIPT%
echo.

if not exist "%PS_SCRIPT%" (
  echo [ERROR] Could not find:
  echo   %PS_SCRIPT%
  echo.
  echo Keep this .cmd file in the repo root.
  echo.
  pause
  exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -RepoRoot "%REPO_ROOT%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo ==> Launcher exited with code %EXIT_CODE%
) else (
  echo ==> Launcher finished successfully
)
echo.
pause
exit /b %EXIT_CODE%
