@echo off
rem Builds the BandSmith app and its installers:
rem   dist\bandsmith\bandsmith.exe          the app, runnable as-is
rem   dist\bandsmith-<version>.msi          the MSI installer
rem   dist\bandsmith-setup-<version>.exe    an .exe installer too, if Inno Setup is installed
rem It installs what it needs to build (Python, the .NET SDK, WiX) with winget if missing.
rem See BUILD.md.
setlocal
cd /d "%~dp0"
if "%BANDSMITH_VERSION%"=="" set BANDSMITH_VERSION=0.0.1

rem ---- build tools --------------------------------------------------------
where py >nul 2>nul || (
  echo Installing Python...
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements || goto :fail
  set "PATH=%LOCALAPPDATA%\Programs\Python\Launcher;%PATH%"
)
where dotnet >nul 2>nul || (
  echo Installing the .NET SDK, needed for WiX...
  winget install -e --id Microsoft.DotNet.SDK.8 --accept-package-agreements --accept-source-agreements || goto :fail
  set "PATH=%ProgramFiles%\dotnet;%PATH%"
)
set "PATH=%USERPROFILE%\.dotnet\tools;%PATH%"
where wix >nul 2>nul || (
  echo Installing WiX 5...
  dotnet tool install --global wix --version 5.0.2 || goto :fail
)
wix extension add -g WixToolset.UI.wixext/5.0.2 >nul || goto :fail

rem ---- the app ------------------------------------------------------------
if not exist .venv (
  py -3 -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
rem the shared core from this checkout, then BandSmith's own requirements
python -m pip install -r ..\requirements.txt ..\..\core pyinstaller || goto :fail

pyinstaller --noconfirm --clean --distpath dist --workpath build bandsmith.spec || goto :fail

rem ---- installers ---------------------------------------------------------
python harvest.py dist\bandsmith files.wxs || goto :fail
wix build bandsmith.wxs files.wxs -arch x64 -ext WixToolset.UI.wixext -d Version=%BANDSMITH_VERSION% -o dist\bandsmith-%BANDSMITH_VERSION%.msi || goto :fail
rem a real build is tens of MB (Python and Qt); a tiny MSI means the app files didn't go in
for %%F in (dist\bandsmith-%BANDSMITH_VERSION%.msi) do if %%~zF LSS 20000000 (
  echo The MSI is only %%~zF bytes, so the app files are missing from it.
  goto :fail
)

set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="%ProgramFiles%\Inno Setup 6\ISCC.exe"
rem BANDSMITH_SKIP_EXE=1 skips it (test builds only need the MSI)
if not "%BANDSMITH_SKIP_EXE%"=="1" if exist %ISCC% %ISCC% installer.iss

echo.
echo Done. Installer: %~dp0dist\bandsmith-%BANDSMITH_VERSION%.msi
exit /b 0

:fail
echo.
echo Build failed (see the error above).
exit /b 1
