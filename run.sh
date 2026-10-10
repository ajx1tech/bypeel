#!/usr/bin/env bash
# bypeel — one-command offline setup + launch (Linux / macOS)
#   ./run.sh                    auto-detects Python 3.11-3.13
#   PYTHON=python3.12 ./run.sh  force a specific interpreter
#   PORT=8001 ./run.sh          use a different port
set -e
cd "$(dirname "$0")"

# ---- 1. Find a supported Python (3.11-3.13, 64-bit) ----------------------
# scikit-learn==1.9.0 (the exact version that pickled the bundled model) ships
# wheels only for Python >= 3.11, and shap==0.48.0 only up to 3.13.
CHECK='import sys,struct; sys.exit(0 if sys.version_info[0]==3 and sys.version_info[1] in (11,12,13) and struct.calcsize("P")==8 else 1)'
PYEXE=""
for cand in "${PYTHON:-}" python3.12 python3.13 python3.11 python3 python; do
  [ -z "$cand" ] && continue
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "$CHECK" >/dev/null 2>&1; then PYEXE="$cand"; break; fi
done
if [ -z "$PYEXE" ]; then
  echo "[bypeel] ERROR: no 64-bit Python 3.11, 3.12 or 3.13 found on this machine."
  echo "[bypeel]        python3 here is: $(python3 --version 2>&1 || echo 'not installed')"
  echo "[bypeel]        Ubuntu/Debian:  sudo apt update && sudo apt install -y python3-venv python3-pip"
  echo "[bypeel]        Ubuntu 22.04 (ships 3.10):  sudo add-apt-repository -y ppa:deadsnakes/ppa &&"
  echo "[bypeel]                        sudo apt install -y python3.12 python3.12-venv"
  echo "[bypeel]        then re-run:    PYTHON=python3.12 ./run.sh"
  exit 1
fi
echo "[bypeel] Using $($PYEXE --version) ($(command -v $PYEXE))"

# ---- 2. Virtual environment (rebuilt automatically if stale/wrong version)
if [ -d ".venv" ] && ! .venv/bin/python -c "$CHECK" >/dev/null 2>&1; then
  echo "[bypeel] Existing .venv is broken or uses an unsupported Python - rebuilding it."
  rm -rf .venv
fi
if [ ! -d ".venv" ]; then
  echo "[bypeel] Creating virtual environment..."
  "$PYEXE" -m venv .venv || { echo "[bypeel] ERROR: venv creation failed. On Ubuntu/Debian run: sudo apt install -y python3-venv"; exit 1; }
fi

# ---- 3. Dependencies: exact pins, pre-built wheels only (never compiles) --
# If a ./wheelhouse folder exists (see README section 17) install fully offline from it.
PIP_ARGS="--only-binary=:all:"
if [ -d "wheelhouse" ]; then
  echo "[bypeel] ./wheelhouse found - installing 100% offline from local wheels..."
  PIP_ARGS="$PIP_ARGS --no-index --find-links wheelhouse"
else
  echo "[bypeel] Installing dependencies (first run needs internet; afterwards 100% offline)..."
  .venv/bin/python -m pip install --quiet --upgrade pip || echo "[bypeel] (pip self-upgrade skipped)"
fi
.venv/bin/python -m pip install --quiet $PIP_ARGS -r requirements.lock.txt || {
  echo "[bypeel] ERROR: dependency install failed - see the message above."
  echo "[bypeel]        No internet on this machine? Use the wheelhouse method in README section 17."
  exit 1; }

# ---- 4. Self-check: model + GeoIP load from local disk -------------------
echo "[bypeel] Verifying model + GeoIP database load correctly..."
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from app.pipeline import get_ml_engine
from app import geoip_lookup
get_ml_engine(); geoip_lookup.lookup_ip('8.8.8.8')
print('  Model + GeoIP OK')
"

PORT="${PORT:-8000}"
echo "[bypeel] Starting server on http://127.0.0.1:$PORT  (Ctrl+C to stop)"
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
