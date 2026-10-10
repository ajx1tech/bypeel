@echo off
REM bypeel - one-command offline setup + launch (Windows 10/11, 64-bit)
REM   run.bat                         auto-detects Python 3.11-3.13
REM   To force an interpreter, set the PYTHON variable first, for example py -3.12
REM   To use another port, set the PORT variable first, for example 8001
cd /d "%~dp0"

REM ---- 1. Find a supported Python (3.11-3.13, 64-bit) ----------------------
REM scikit-learn==1.9.0 (the exact version that pickled the bundled model) ships
REM wheels only for Python 3.11 and newer, and shap==0.48.0 only up to 3.13.
set "CHECK=import sys,struct; sys.exit(0 if sys.version_info[0]==3 and sys.version_info[1] in (11,12,13) and struct.calcsize('P')==8 else 1)"
set "PYEXE="
for %%C in ("%PYTHON%" python "py -3.12" "py -3.13" "py -3.11") do (
  if not "%%~C"=="" if not defined PYEXE (
    %%~C -c "%CHECK%" >nul 2>nul
    if not errorlevel 1 set "PYEXE=%%~C"
  )
)
if not defined PYEXE (
  echo [bypeel] ERROR: no 64-bit Python 3.11, 3.12 or 3.13 was found.
  echo [bypeel]   1. Install it:  winget install -e --id Python.Python.3.12
  echo [bypeel]      or download the 64-bit installer from https://www.python.org/downloads/windows/
  echo [bypeel]   2. During install tick "Add python.exe to PATH".
  echo [bypeel]   3. Open a NEW terminal and run this script again.
  echo [bypeel]   Python 3.10 or older, 3.14, and 32-bit Python are not supported.
  pause
  exit /b 1
)
echo [bypeel] Using: %PYEXE%

REM ---- 2. Virtual environment (rebuilt automatically if stale) --------------
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "%CHECK%" >nul 2>nul
  if errorlevel 1 (
    echo [bypeel] Existing .venv is broken or uses an unsupported Python - rebuilding it.
    rmdir /s /q .venv
  )
)
if not exist ".venv\Scripts\python.exe" (
  echo [bypeel] Creating virtual environment...
  %PYEXE% -m venv .venv
  if errorlevel 1 (
    echo [bypeel] ERROR: could not create the virtual environment.
    pause
    exit /b 1
  )
)

REM ---- 3. Dependencies: exact pins, pre-built wheels only (never compiles) --
REM If a .\wheelhouse folder exists (see README section 17) install fully offline from it.
set "PIP_ARGS=--only-binary=:all:"
if exist "wheelhouse" (
  echo [bypeel] wheelhouse folder found - installing 100%% offline from local wheels...
  set "PIP_ARGS=--only-binary=:all: --no-index --find-links wheelhouse"
) else (
  echo [bypeel] Installing dependencies - first run needs internet, afterwards 100%% offline...
  ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
)
".venv\Scripts\python.exe" -m pip install --quiet %PIP_ARGS% -r requirements.lock.txt
if errorlevel 1 (
  echo [bypeel] ERROR: dependency install failed - see the message above.
  echo [bypeel]   No internet on this machine? Use the wheelhouse method in README section 17.
  echo [bypeel]   Behind a proxy? Set HTTPS_PROXY first, see README section 18.
  pause
  exit /b 1
)

REM ---- 4. Self-check: model + GeoIP load from local disk -------------------
echo [bypeel] Verifying model + GeoIP database load correctly...
".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0,'.'); from app.pipeline import get_ml_engine; from app import geoip_lookup; get_ml_engine(); geoip_lookup.lookup_ip('8.8.8.8'); print('  Model + GeoIP OK')"
if errorlevel 1 (
  echo [bypeel] ERROR: the model or GeoIP self-check failed - see the message above.
  pause
  exit /b 1
)

if not defined PORT set "PORT=8000"
echo [bypeel] Starting server on http://127.0.0.1:%PORT%  (Ctrl+C to stop)
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
