# SafeSight PPE Safety

GPU-ready PPE compliance monitoring for CCTV/video streams.

This active workspace uses the SafeSight project structure as the main project. Earlier experimental work was moved to `_retired/safesight_20260604-172036` so the active root stays clean while still being recoverable.

## Current Direction

The project is prepared around real model/runtime behavior:

- Frontend: Next.js dashboard, live/video monitoring, API-only data flow
- Backend: FastAPI, SQLite, video upload/processing, event persistence
- Vision runtime: Ultralytics person tracking + custom PPE YOLO + optional SAM2 masks
- Default PPE profile: construction PPE (`helmet`, `vest`, `gloves`, `boots`, `safety goggles`)
- Mock/demo frontend data: removed
- Backend fake detections: disabled by default
- Current PPE model: available at `backend\weights\ppe_detector\best.pt`
- Missing PPE logic: inferred from required PPE absence, with weak `no_*` classes used only as supporting signals

## Setup

From `D:\safe-sight`:

```powershell
.\scripts\prepare_weights.ps1
.\scripts\install_backend_gpu.ps1
.\scripts\install_frontend.ps1
.\scripts\check_gpu.ps1
```

The backend installer creates `.venv-backend` and installs CUDA PyTorch wheels. It defaults to `cu130`; if your machine needs another CUDA wheel index, rerun with:

```powershell
.\scripts\install_backend_gpu.ps1 -CudaIndex cu128
```

## Required Weights

The app will not fake PPE detections if the model is missing. Place the trained/custom PPE model here:

```text
backend\weights\ppe_detector\best.pt
```

Person detection uses `PERSON_MODEL_NAME=yolo11m.pt` and can auto-download through Ultralytics on first run.

SAM2 is optional. If you want masks, run:

```powershell
.\scripts\prepare_weights.ps1 -DownloadSAM2
.\scripts\install_backend_gpu.ps1 -WithSAM2
```

Without SAM2, the system still associates PPE to person boxes. Box-mask fallback is opt-in only via `ALLOW_BOX_MASK_FALLBACK=true`.

Current downloaded initial assets:

- `backend\weights\person_detector\yolo11m.pt`
- `backend\weights\base\yolo11s.pt`
- `backend\weights\sam2\sam2.1_hiera_base_plus.pt`
- InsightFace `buffalo_l` in `%USERPROFILE%\.insightface\models\buffalo_l`
- Construction-PPE dataset in `data\ppe_construction`
- Cleaned presence-only dataset in `data\ppe_construction_presence`

## Run

Start backend:

```powershell
.\scripts\run_backend.ps1
```

Start frontend:

```powershell
.\scripts\run_frontend.ps1
```

Open:

- UI: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health` returns `ml_ready=false` until `backend\weights\ppe_detector\best.pt` exists.

## Training

The first 11-class model trained successfully, but the weak `no_*` classes limit useful violation accuracy. The preferred next model is presence-only: `helmet`, `gloves`, `vest`, `boots`, `goggles`, and `Person`. Missing PPE is inferred per person at runtime.

Build/check the cleaned dataset:

```powershell
D:\safe-sight\.venv-backend\Scripts\python.exe .\scripts\prepare_ppe_presence_dataset.py
```

Training is intentionally guarded. This command only prints the preferred training plan:

```powershell
.\scripts\train_ppe_presence_yolo11.ps1
```

When you explicitly approve presence-only training:

```powershell
.\scripts\train_ppe_presence_yolo11.ps1 -GreenLight
```

Laptop-safe fallback:

```powershell
.\scripts\train_ppe_presence_yolo11.ps1 -GreenLight -ImgSz 640 -Batch 2 -Workers 0 -CpuThreads 2
```

The older 11-class training script is still available:

```powershell
.\scripts\train_ppe_yolo11.ps1
```

When you explicitly approve training:

```powershell
.\scripts\train_ppe_yolo11.ps1 -GreenLight
```

## Dataset Profiles

Construction profile, active default:

```env
PPE_CLASS_PROFILE=construction
PPE_PROMPTS=["helmet","vest","gloves","boots","safety goggles"]
REQUIRED_PPE=["helmet","vest","safety goggles"]
```

Lab profile, only if using the upstream lab-safety model:

```env
PPE_CLASS_PROFILE=lab
PPE_PROMPTS=["safety goggles","face mask","lab coat","gloves","head mask"]
REQUIRED_PPE=["safety goggles","face mask","lab coat"]
```

Configs are in `backend\configs`.

## Project Structure

```text
D:\safe-sight
├── backend\        FastAPI app, ML pipeline, model configs
├── frontend\       Next.js dashboard
├── data\           local datasets, videos, outputs, snapshots
├── scripts\        setup/run/check/train-prep scripts
└── _retired\       archived previous project work
```

## What Is Still Left

- Put the construction PPE dataset into `data\ppe_construction` or let Ultralytics download it during training.
- Install dependencies on the machine where training/runtime will happen.
- Train or provide `backend\weights\ppe_detector\best.pt`.
- Optionally install/download SAM2 if we want mask-level association.
- Run an end-to-end video through the backend after the PPE weight exists.
