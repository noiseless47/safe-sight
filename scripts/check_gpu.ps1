$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv-backend\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Backend venv not found. Run scripts/install_backend_gpu.ps1 first."
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi
}
else {
  Write-Warning "nvidia-smi was not found on PATH."
}

$check = @'
import importlib
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram gb:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

for package in ["ultralytics", "cv2", "fastapi", "onnxruntime"]:
    mod = importlib.import_module(package)
    version = getattr(mod, "__version__", "installed")
    print(f"{package}: {version}")
'@

$check | & $Python -
if ($LASTEXITCODE -ne 0) {
  throw "GPU dependency check failed with exit code $LASTEXITCODE"
}
