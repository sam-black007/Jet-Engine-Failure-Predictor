@echo off
title JETWISE AI - Turbofan Simulation Platform
echo =====================================================================
echo    JETWISE AI - Enterprise Turbofan Simulation and Digital Twin
echo =====================================================================
echo.

cd /d "%~dp0"

echo [1/2] Verifying Python and MATLAB Engine API...
python -c "from dotenv import load_dotenv; load_dotenv('.env'); from src.matlab_simulink_bridge import MATLABSimulinkBridge; print('MATLAB Status:', MATLABSimulinkBridge().get_status()['message'])"

echo.
echo [2/2] Launching JETWISE AI Streamlit Dashboard...
echo Opening http://localhost:8501 in default browser...
start "" http://localhost:8501
python -m streamlit run dashboard/app.py

pause
