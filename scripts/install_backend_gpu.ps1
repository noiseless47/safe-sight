param(
  [string]$Python = "",
  [string]$CudaIndex = "cu130",
  [switch]$CpuOnly,
  [switch]$WithSAM2
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $Root ".venv-backend"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Resolve-PythonCommand {
  param([string]$RequestedPython)

  if ($RequestedPython) {
    return @($RequestedPython)
  }

  $candidatePaths = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
  )

  foreach ($candidate in $candidatePaths) {
    if (Test-Path -LiteralPath $candidate) {
      return @($candidate)
    }
  }

  if (Get-Command py -ErrorAction SilentlyContinue) {
    return @("py", "-3.11")
  }

  if (Get-Command python -ErrorAction SilentlyContinue) {
    return @("python")
  }

  throw "Python 3.11 or 3.12 was not found. Install Python 3.11 first."
}

function Invoke-Python {
  param(
    [string[]]$PythonCommand,
    [string[]]$Arguments
  )

  $exe = $PythonCommand[0]
  $prefixArgs = @()
  if ($PythonCommand.Count -gt 1) {
    $prefixArgs = $PythonCommand[1..($PythonCommand.Count - 1)]
  }

  & $exe @prefixArgs @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$exe failed with exit code $LASTEXITCODE"
  }
}

function Invoke-Checked {
  param(
    [string]$FilePath,
    [string[]]$Arguments
  )

  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$FilePath failed with exit code $LASTEXITCODE"
  }
}

$pythonCommand = Resolve-PythonCommand -RequestedPython $Python

if (-not (Test-Path -LiteralPath $VenvPython)) {
  Write-Host "Creating backend venv at $VenvDir"
  Invoke-Python -PythonCommand $pythonCommand -Arguments @("-m", "venv", $VenvDir)
}

Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

if ($CpuOnly) {
  Write-Host "Installing CPU PyTorch wheels"
  Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "torch", "torchvision", "torchaudio")
}
else {
  $indexUrl = "https://download.pytorch.org/whl/$CudaIndex"
  Write-Host "Installing PyTorch CUDA wheels from $indexUrl"
  Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "torch", "torchvision", "torchaudio", "--index-url", $indexUrl)
}

Write-Host "Installing backend package and ML dependencies"
$BackendDir = Join-Path $Root "backend"
Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "-e", $BackendDir)
Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "setuptools<82")

if ($WithSAM2) {
  Write-Host "Installing SAM2 from Meta repository"
  Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "git+https://github.com/facebookresearch/sam2.git")
}

Write-Host ""
Write-Host "Backend GPU environment prepared. No training was started."
Write-Host "Run scripts/check_gpu.ps1 next."
