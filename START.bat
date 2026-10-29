@echo off
title TomatoAI Launcher
color 0A

echo.
echo  ============================================
echo   🍅  TomatoAI - Starting Up...
echo  ============================================
echo.

:: Change to project directory
cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    if not exist "venv\python.exe" (
        echo  [ERROR] Virtual environment not found!
        echo  Please run: python -m venv venv
        echo  Then: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

:: Check if cloudflared exists
if not exist "tools\cloudflared.exe" (
    echo  [WARNING] cloudflared.exe not found in tools\
    echo  Public URL will NOT be generated.
    echo  Only localhost will be started.
    echo.
    set CF_AVAILABLE=0
) else (
    set CF_AVAILABLE=1
)

echo  [1/3] Starting FastAPI server on http://localhost:8000 ...
echo.

:: Start uvicorn in a new window
start "TomatoAI - FastAPI Server" cmd /k "cd /d "%~dp0" && color 0A && echo  [FastAPI] Starting server... && venv\Scripts\python.exe -m uvicorn app.server:app --host 0.0.0.0 --port 8000 2>nul || venv\python.exe -m uvicorn app.server:app --host 0.0.0.0 --port 8000"

echo  [2/3] Waiting for server to be ready (5 seconds)...
timeout /t 5 /nobreak >nul

:: Open browser automatically
echo  [3/3] Opening browser at http://localhost:8000 ...
start "" "http://localhost:8000"

if "%CF_AVAILABLE%"=="1" (
    echo.
    echo  Starting Cloudflare Tunnel for public URL...
    echo  Look at the new window for your public link!
    echo.
    start "TomatoAI - Cloudflare Tunnel" cmd /k "cd /d "%~dp0" && color 09 && echo. && echo  ============================================ && echo   Copy the trycloudflare.com link below! && echo  ============================================ && echo. && tools\cloudflared.exe tunnel --url http://localhost:8000"
)

echo.
echo  ============================================
echo   TomatoAI is running!
echo.
echo   Local:   http://localhost:8000
if "%CF_AVAILABLE%"=="1" (
echo   Public:  Check the blue Cloudflare window
)
echo.
echo   Close the FastAPI window to stop the app.
echo  ============================================
echo.
pause
