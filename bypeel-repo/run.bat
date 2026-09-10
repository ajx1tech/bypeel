@echo off
REM bypeel -- one-command offline setup + launch (Windows)
cd /d "%~dp0"

if not exist ".venv" (
  echo [bypeel] Creating virtual environment...
  python -m venv .venv
)

call .venv\Scripts\activate.bat

echo [bypeel] Installing dependencies (first run only, offline after this)...
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo [bypeel] Verifying model + GeoIP database load correctly...
python -c "import sys; sys.path.insert(0,'.'); from app.pipeline import get_ml_engine; from app import geoip_lookup; get_ml_engine(); geoip_lookup.lookup_ip('8.8.8.8'); print('  Model + GeoIP OK')"

echo [bypeel] Starting server on http://127.0.0.1:8000 (Ctrl+C to stop)...
uvicorn app.main:app --host 127.0.0.1 --port 8000
