@echo off
title Instagator - Streamlit App
echo Launching Instagator Streamlit app...
echo.


REM Navigate to project directory
cd /d "%~dp0"

REM Run Streamlit app
streamlit run main.py --server.port 8501 --server.address 127.0.0.1

pause
