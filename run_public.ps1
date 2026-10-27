# =============================================================================
# run_public.ps1 — Start the FastAPI server + Cloudflare Public HTTPS Tunnel
# =============================================================================

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "🍅 Starting Tomato Disease Intelligence (Public Deployment)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# 1. Start FastAPI server in background if not already running on port 8000
$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if (-not $port8000) {
    Write-Host "Starting FastAPI web server on http://localhost:8000..." -ForegroundColor Cyan
    Start-Process -FilePath ".\venv\python.exe" -ArgumentList "-m uvicorn app.server:app --host 0.0.0.0 --port 8000" -WindowStyle Minimized
    Start-Sleep -Seconds 3
} else {
    Write-Host "FastAPI server is already running on port 8000." -ForegroundColor Cyan
}

# 2. Launch Cloudflare Tunnel
Write-Host "Launching Cloudflare Tunnel to expose the app globally..." -ForegroundColor Yellow
Write-Host "Look for the public HTTPS link below (e.g., https://xxxx.trycloudflare.com):" -ForegroundColor White
Write-Host "Press Ctrl+C to stop the public tunnel at any time.`n" -ForegroundColor Gray

.\tools\cloudflared.exe tunnel --url http://localhost:8000
