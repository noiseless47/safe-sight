param(
  [switch]$SkipConstructionActivity
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv-backend\Scripts\python.exe"
$Target = Join-Path $Root "data\videos\validation\kaggle"

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Backend venv not found. Run scripts/install_backend_gpu.ps1 first."
}

try {
  & $Python -m kaggle --version | Out-Null
}
catch {
  & $Python -m pip install kaggle
}

New-Item -ItemType Directory -Force -Path $Target | Out-Null

$datasets = @(
  @{
    Ref = "hinepo/video-example-for-ppe-red-zone"
    Dir = "ppe_red_zone"
  },
  @{
    Ref = "slimese/ppe-video"
    Dir = "ppe_video"
  }
)

if (-not $SkipConstructionActivity) {
  $datasets += @{
    Ref = "ehsaanali/construction-activity-recognition-dataset"
    Dir = "construction_activity"
  }
}

foreach ($dataset in $datasets) {
  $OutDir = Join-Path $Target $dataset.Dir
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
  & $Python -m kaggle datasets download $dataset.Ref -p $OutDir --unzip
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to download $($dataset.Ref)"
  }
}

$videos = Get-ChildItem -Path $Target -Recurse -File -Include *.mp4,*.avi,*.mov,*.mkv,*.webm
$totalMb = ($videos | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "Downloaded validation videos: $($videos.Count)"
Write-Host ("Total size: {0:N2} MB" -f $totalMb)
Write-Host "Location: $Target"
