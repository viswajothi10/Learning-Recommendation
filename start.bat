@echo off
echo ============================================
echo   AI Learning Recommendation System
echo ============================================
echo.

echo [1/2] Starting Flask Backend on http://localhost:5000
start "Flask Backend" cmd /k "cd /d "%~dp0backend" && pip install -r requirements.txt -q && python app.py"

timeout /t 4 /nobreak >nul

echo [2/2] Starting React Frontend on http://localhost:3000
start "React Frontend" cmd /k "cd /d "%~dp0frontend" && npm install && npm start"

echo.
echo Both servers are starting...
echo Backend  → http://localhost:5000
echo Frontend → http://localhost:3000
echo.
pause
