@echo off
title AeroLens AQI Intelligence UI
cd /d "%~dp0"
echo ========================================================
echo       Starting AeroLens AQI Intelligence Web App        
echo ========================================================
echo.
echo Opening http://127.0.0.1:5000 in your browser...
start http://127.0.0.1:5000
echo.
python app.py
pause

