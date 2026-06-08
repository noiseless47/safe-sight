# Project Status

## Done

- Previous SafeSight implementation archived under `_retired/safesight_20260604-172036`.
- Downloaded SentinelVision code promoted into the active project root.
- Frontend demo fixtures, demo videos, and simulated query paths removed.
- Backend mock detections disabled by default.
- PPE detector now supports explicit `construction` and `lab` class profiles.
- Construction PPE is the active default profile.
- Python backend dependency metadata updated for Python 3.11/3.12 and CUDA PyTorch wheels.
- GPU/backend/frontend/setup/run scripts added under `scripts`.
- Training script added with a `-GreenLight` guard so training cannot start accidentally.
- Dataset YAMLs added under `backend/configs`.
- PPE runtime now infers missing PPE from absence of required positive PPE detections, instead of depending only on weak `no_*` detector classes.
- YOLO class roles are derived from `model.names`, so both the original 11-class model and a cleaned presence-only model can be loaded.
- Presence-only dataset builder added at `scripts/prepare_ppe_presence_dataset.py`.
- Presence-only training script added at `scripts/train_ppe_presence_yolo11.ps1`.
- Initial model assets downloaded:
  - `backend/weights/person_detector/yolo11m.pt`
  - `backend/weights/base/yolo11s.pt`
  - `backend/weights/sam2/sam2.1_hiera_base_plus.pt`
  - InsightFace `buffalo_l` under the user model cache.
- SAM2 Python package installed in `.venv-backend`.
- Construction-PPE dataset downloaded and validated at `data/ppe_construction`.
- Cleaned presence-only Construction-PPE dataset generated at `data/ppe_construction_presence`.
- Current PPE model exists at `backend/weights/ppe_detector/best.pt`.

## Not Done Yet

- Original 11-class PPE training plateaued around `0.60` validation mAP50 and `0.54` test mAP50 because the `no_*` classes are sparse and weak.
- Presence-only model has not been trained yet.
- End-to-end video validation still needs to be repeated with the runtime absence-inference fix.

## Next Stage

1. Restart the backend so it loads the absence-inference runtime changes.
2. Validate current `backend/weights/ppe_detector/best.pt` on a real video.
3. For the stronger model path, run `scripts/train_ppe_presence_yolo11.ps1 -GreenLight`.
4. Validate the presence-only model on video and tune `REQUIRED_PPE`/thresholds.
