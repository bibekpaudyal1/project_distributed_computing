@echo off
REM Start a LocalMesh AI node on Windows.  start.bat [--demo] [--token SECRET]
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.9+ is required: https://www.python.org/downloads/ & exit /b 1)
if not exist .venv (
  echo First run: creating a virtual environment
  python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -q --disable-pip-version-check -r requirements.txt
python -m localmesh %*
