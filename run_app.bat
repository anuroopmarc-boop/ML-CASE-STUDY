@echo off
title 23CSE301 ML Case Study - Streamlit GUI
echo ========================================================
echo Starting ML Case Study Web Application...
echo ========================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
