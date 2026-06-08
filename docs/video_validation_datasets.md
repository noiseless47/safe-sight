# Video Validation Datasets

Bulk-downloaded validation videos are stored under:

```text
data/videos/validation/kaggle
```

## Included Sources

| Dataset | Kaggle ref | Purpose | License note |
| --- | --- | --- | --- |
| PPE red-zone example video | `hinepo/video-example-for-ppe-red-zone` | PPE/red-zone validation clip | Kaggle reports unknown license |
| PPE video | `slimese/ppe-video` | PPE validation clips | CC0-1.0 |
| Construction activity recognition dataset | `ehsaanali/construction-activity-recognition-dataset` | Construction-video stress/background validation | CC0-1.0 |

## Download

```powershell
.\scripts\download_validation_videos.ps1
```

To download only the PPE-specific videos:

```powershell
.\scripts\download_validation_videos.ps1 -SkipConstructionActivity
```

## Current Pack

The current downloaded pack contains 88 videos, about 148 MB total.

Important limitation: this is a bulk validation video pack, not a fully annotated CCTV PPE benchmark. Use it to test runtime behavior, threshold tuning, false positives, and visual output. Keep official model metrics tied to the labeled image datasets unless we manually annotate video frames.
