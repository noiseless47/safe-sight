param(
  [switch]$GreenLight,
  [string]$Data = "backend\configs\ppe_construction_yolo11.yaml",
  [string]$Model = "backend\weights\base\yolo11s.pt",
  [int]$Epochs = 120,
  [int]$ImgSz = 640,
  [string]$Device = "0",
  [int]$Batch = 2,
  [switch]$Resume,
  [int]$Workers = 0,
  [int]$CpuThreads = 2
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv-backend\Scripts\python.exe"
$DataPath = (Resolve-Path -LiteralPath (Join-Path $Root $Data) -ErrorAction Stop).Path
$ModelPath = (Resolve-Path -LiteralPath (Join-Path $Root $Model) -ErrorAction Stop).Path
$ProjectDir = Join-Path $Root "backend\runs\ppe"
$TargetWeights = Join-Path $Root "backend\weights\ppe_detector\best.pt"
$RunWeights = Join-Path $ProjectDir "construction_yolo11\weights"
$ResumeCheckpoint = Join-Path $RunWeights "last.pt"
$EffectiveModelPath = $ModelPath
$ResumeValue = "False"

$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
$env:NUMEXPR_NUM_THREADS = "$CpuThreads"

if ($Resume) {
  if (-not (Test-Path -LiteralPath $ResumeCheckpoint)) {
    throw "Resume requested, but no checkpoint exists at $ResumeCheckpoint"
  }

  $EffectiveModelPath = (Resolve-Path -LiteralPath $ResumeCheckpoint -ErrorAction Stop).Path
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

if (-not $GreenLight) {
  Write-Host "Training is prepared but not started."
  Write-Host "When you approve training, run:"
  Write-Host "  .\scripts\train_ppe_yolo11.ps1 -GreenLight"
  Write-Host ""
  Write-Host "Continue from previous weights with safe settings:"
  Write-Host "  .\scripts\train_ppe_yolo11.ps1 -GreenLight -Resume"
  Write-Host ""
  Write-Host "Defaults: model=$Model epochs=$Epochs imgsz=$ImgSz batch=$Batch device=$Device workers=$Workers cpu_threads=$CpuThreads data=$Data"
  exit 0
}

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Backend venv not found. Run scripts/install_backend_gpu.ps1 first."
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $TargetWeights) | Out-Null
New-Item -ItemType Directory -Force -Path $ProjectDir | Out-Null
$LauncherPath = Join-Path $ProjectDir "train_launcher.py"

$train = @"
import shutil
import time

import torch
from ultralytics import YOLO

torch.set_num_threads($CpuThreads)
torch.set_num_interop_threads(1)


def patch_ultralytics_checkpoint_saver():
    import ultralytics.engine.trainer as trainer_mod

    def save_model(self):
        ema = trainer_mod.deepcopy(trainer_mod.unwrap_model(self.ema.ema)).half()
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            trainer_mod.LOGGER.warning(f"Skipping checkpoint save at epoch {self.epoch}: EMA contains NaN/Inf")
            return False

        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,
            "ema": ema,
            "updates": self.ema.updates,
            "optimizer": trainer_mod.convert_optimizer_state_dict_to_fp16(
                trainer_mod.deepcopy(self.optimizer.state_dict())
            ),
            "scaler": self.scaler.state_dict(),
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
            "train_results": self.read_results_csv(),
            "date": trainer_mod.datetime.now().isoformat(),
            "version": trainer_mod.__version__,
            "git": {
                "root": str(trainer_mod.GIT.root),
                "branch": trainer_mod.GIT.branch,
                "commit": trainer_mod.GIT.commit,
                "message": trainer_mod.GIT.message,
                "origin": trainer_mod.GIT.origin,
            },
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }

        self.wdir.mkdir(parents=True, exist_ok=True)
        tmp = self.wdir / f".{self.last.stem}.tmp.pt"

        for attempt in range(4):
            try:
                torch.save(ckpt, str(tmp))
                tmp.replace(self.last)
                break
            except (RuntimeError, OSError, ValueError):
                if tmp.exists():
                    tmp.unlink()
                if attempt == 3:
                    raise
                time.sleep((2**attempt) / 2)

        if self.best_fitness == self.fitness:
            shutil.copyfile(self.last, self.best)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            shutil.copyfile(self.last, self.wdir / f"epoch{self.epoch}.pt")
        return True

    trainer_mod.BaseTrainer.save_model = save_model


def main():
    patch_ultralytics_checkpoint_saver()
    model = YOLO(r"""$EffectiveModelPath""")
    model.train(
        data=r"""$DataPath""",
        epochs=$Epochs,
        imgsz=$ImgSz,
        device=r"""$Device""",
        project=r"""$ProjectDir""",
        name="construction_yolo11",
        exist_ok=True,
        batch=$Batch,
        amp=True,
        workers=$Workers,
        patience=40,
        cos_lr=True,
        close_mosaic=15,
        plots=True,
        save_json=True,
        resume=$ResumeValue,
    )

if __name__ == "__main__":
    main()
"@

Set-Content -LiteralPath $LauncherPath -Value $train -Encoding UTF8

Push-Location $Root
try {
  Invoke-Checked -FilePath $Python -Arguments @($LauncherPath)
}
finally {
  Pop-Location
}

$BestWeights = Join-Path $ProjectDir "construction_yolo11\weights\best.pt"
if (Test-Path -LiteralPath $BestWeights) {
  Copy-Item -LiteralPath $BestWeights -Destination $TargetWeights -Force
  Write-Host "Best PPE weights copied to backend\weights\ppe_detector\best.pt"
}
else {
  Write-Warning "Training finished, but best.pt was not found at $BestWeights"
}
