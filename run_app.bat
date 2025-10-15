@echo off
title Instagator - Streamlit App
echo Launching Instagator Streamlit app...
echo.

REM (Optional) Activate virtual environment if you use one:
REM call venv\Scripts\activate

REM Navigate to project directory
cd /d "%~dp0"

REM Run Streamlit app
streamlit run main.py --server.port 8501 --server.address 127.0.0.1

pause
