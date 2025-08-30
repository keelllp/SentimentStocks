@echo off
echo ========================================
echo   StockPredict AI Production System
echo ========================================
echo.
echo Starting Flask Backend and React Frontend...
echo.

REM Start Flask Backend in a new window
start "Flask Backend" cmd /k "cd backend && python app.py"

REM Wait a moment for Flask to start
timeout /t 3 /nobreak > nul

REM Start React Frontend in a new window
start "React Frontend" cmd /k "cd frontend && npm start"

echo.
echo ========================================
echo   Both services are starting...
echo ========================================
echo.
echo Flask Backend: http://localhost:5000
echo React Frontend: http://localhost:3000
echo.
echo Press any key to close this window...
pause > nul
