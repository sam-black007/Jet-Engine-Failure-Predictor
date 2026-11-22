@echo off
title THRUSTLAB - Integrated Turbofan Digital Twin & PHM Platform
echo =========================================================================================
echo    THRUSTLAB: Integrated AI-Driven and Physics-Informed Digital Twin Framework
echo =========================================================================================
echo.

cd /d "%~dp0"

echo [1/2] Verifying Python and MATLAB Engine API...
python -c "from dotenv import load_dotenv; load_dotenv('.env'); from src.matlab_simulink_bridge import MATLABSimulinkBridge; print('MATLAB Status:', MATLABSimulinkBridge().get_status()['message'])"

echo.
echo [2/2] Launching THRUSTLAB Streamlit Dashboard...
echo Opening http://localhost:8501 in default browser...
start "" http://localhost:8501
python -m streamlit run dashboard/app.py

pause
