$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv-backend\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Backend venv not found. Run scripts/install_backend_gpu.ps1 first."
}

Push-Location $Root
try {
  & $Python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
}
finally {
  Pop-Location
}
