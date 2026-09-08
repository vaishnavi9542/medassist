# start-dev.ps1
# Run from the project root to start both the backend API and the frontend dev server.
# Usage: .\start-dev.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPath = Join-Path $root 'backend'
$frontendPath = Join-Path $backendPath 'frontend'
$venvPython = Join-Path $root 'venv\Scripts\python.exe'
$pyLauncher = 'py.exe'

if (Test-Path $venvPython) {
    $pythonPath = $venvPython
} elseif (Get-Command $pyLauncher -ErrorAction SilentlyContinue) {
    $pythonPath = $pyLauncher
} else {
    Write-Error "Python executable not found. Ensure your virtual environment exists at '$venvPython' or that 'py' is available on PATH."
    exit 1
}

Write-Host 'Starting backend server on http://127.0.0.1:8001 ...'
Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoExit', '-Command', "Set-Location -LiteralPath '$backendPath'; & '$pythonPath' -m db.init_db; if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }; & '$pythonPath' -m uvicorn app.main:app --reload --port 8001"

Write-Host 'Starting frontend server on http://127.0.0.1:5173 ...'
Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoExit', '-Command', "Set-Location -LiteralPath '$frontendPath'; npm install; npm run dev"

Write-Host 'Both servers have been launched in separate PowerShell windows.'
Write-Host 'Wait for uvicorn and Vite logs to appear in the new windows before using the app.'
