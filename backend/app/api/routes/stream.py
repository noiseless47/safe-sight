"""
Video upload and processing endpoints for recorded video analysis.
Includes annotated video generation for frontend display.
"""

import asyncio
import cv2
import numpy as np
import logging
import subprocess
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid
from pydantic import BaseModel

from ...ml.pipeline import get_pipeline

logger = logging.getLogger(__name__)


def make_browser_playable_mp4(video_path: Path) -> Path:
    """Transcode OpenCV MP4 output to H.264 so browser video playback is reliable."""
    try:
        import imageio_ffmpeg
    except ImportError:
        logger.warning("imageio-ffmpeg missing; serving OpenCV MP4 output without H.264 transcode")
        return video_path

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    transcode_path = video_path.with_name(f"{video_path.stem}_browser.mp4")
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(transcode_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if transcode_path.exists() and transcode_path.stat().st_size > 0:
            transcode_path.replace(video_path)
    except Exception as exc:
        logger.warning("Failed to transcode %s for browser playback: %s", video_path, exc)
        if transcode_path.exists():
            transcode_path.unlink()

    return video_path


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class ProcessVideoRequest(BaseModel):
    video_path: str


from ...core.config import settings
from ...core.database import async_session
from ...services.persistence import PersistenceManager


router = APIRouter(prefix="/stream", tags=["video"])
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _resolve_video_path(video_path: str) -> Path:
    """Resolve a video path while keeping access inside the configured video dir."""
    settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    root = settings.VIDEOS_DIR.resolve()
    candidate = Path(video_path)
    if not candidate.is_absolute():
        candidate = settings.VIDEOS_DIR / video_path

    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Video path must be inside {settings.VIDEOS_DIR}",
        ) from exc

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    if resolved.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    return resolved


class VideoProcessingManager:
    """Manages video processing jobs."""

    ACTIVE_STATUSES = {"pending", "processing"}

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, video_path: str, filename: str) -> str:
        """Create a new processing job."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "video_path": str(video_path),
            "filename": str(filename),
            "status": "pending",
            "progress": 0,
            "total_frames": 0,
            "processed_frames": 0,
            "violations_count": 0,
            "persons_count": 0,
            "unique_events": 0,  # Deduplicated event count
            "closed_events": 0,
            "error": None,
            "output_video": None,  # Path to annotated video
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        job = self.jobs.get(job_id)
        if job:
            return convert_numpy_types(job)
        return None

    def update_job(self, job_id: str, **kwargs):
        """Update job status."""
        if job_id in self.jobs:
            # Convert numpy types to native Python types
            converted_kwargs = convert_numpy_types(kwargs)
            self.jobs[job_id].update(converted_kwargs)

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs."""
        return [convert_numpy_types(job) for job in self.jobs.values()]

    def find_active_job(self) -> Optional[Dict[str, Any]]:
        """Return the newest active video processing job, if one exists."""
        for job in reversed(list(self.jobs.values())):
            if job.get("status") in self.ACTIVE_STATUSES:
                return convert_numpy_types(job)
        return None

    def find_active_job_for_video(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Return an active job for the same resolved video path, if one exists."""
        resolved_path = str(Path(video_path).resolve())
        for job in reversed(list(self.jobs.values())):
            if (
                job.get("status") in self.ACTIVE_STATUSES
                and str(Path(job.get("video_path", "")).resolve()) == resolved_path
            ):
                return convert_numpy_types(job)
        return None


processing_manager = VideoProcessingManager()


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing."""
    if Path(file.filename).suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid video format. Supported: mp4, avi, mov, mkv, webm",
        )

    # Ensure videos directory exists
    settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Save to videos directory
    video_path = settings.VIDEOS_DIR / file.filename

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return JSONResponse(
        {
            "message": "Video uploaded successfully",
            "filename": file.filename,
            "path": str(video_path),
        }
    )


@router.post("/process")
async def start_video_processing(
    request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start processing an uploaded video.

    Returns a job_id that can be used to track progress.
    """
    path = _resolve_video_path(request.video_path)

    active_job = processing_manager.find_active_job_for_video(str(path))
    if active_job:
        return {
            "message": "Processing already active",
            "job_id": active_job["id"],
            "video_path": str(path),
            "reused": True,
        }

    other_active_job = processing_manager.find_active_job()
    if other_active_job:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Another analysis is already running for "
                f"{other_active_job.get('filename', 'a video')}. "
                "Wait for it to finish before starting a new video."
            ),
        )

    # Create processing job
    job_id = processing_manager.create_job(str(path), path.name)

    # Start background processing
    background_tasks.add_task(process_video_task, job_id, str(path))

    return {
        "message": "Processing started",
        "job_id": job_id,
        "video_path": str(path),
        "reused": False,
    }


async def process_video_task(job_id: str, video_path: str):
    """
    Background task to process a video file.

    Creates an annotated output video and uses event deduplication.
    """
    processing_manager.update_job(job_id, status="processing")

    pipeline = get_pipeline()
    pipeline.initialize()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        processing_manager.update_job(
            job_id, status="failed", error=f"Could not open video: {video_path}"
        )
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_skip = max(1, int(video_fps / settings.FRAME_SAMPLE_RATE))

    processing_manager.update_job(job_id, total_frames=total_frames)

    cap.release()

    # Setup output video writer for annotated frames
    output_dir = settings.PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    input_filename = Path(video_path).stem
    output_filename = f"{input_filename}_annotated_{job_id[:8]}.mp4"
    output_path = output_dir / output_filename

    # OpenCV writes an intermediate MP4; it is transcoded to browser-safe H.264 at completion.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_fps = min(settings.FRAME_SAMPLE_RATE, video_fps)
    video_writer = cv2.VideoWriter(
        str(output_path), fourcc, output_fps, (frame_width, frame_height)
    )

    processed_count = 0
    total_unique_events = 0
    unique_persons = set()

    try:
        for result in pipeline.process_video(video_path):
            processed_count += 1

            # Add video_source to result for persistence
            result["video_source"] = video_path

            # Write annotated frame to output video
            annotated_frame = result.get("annotated_frame")
            if annotated_frame is not None and video_writer.isOpened():
                video_writer.write(annotated_frame)

            # Persist with deduplication - process all persons, not just events
            async with async_session() as session:
                persistence = PersistenceManager(session)
                persist_result = await persistence.persist_frame_results(
                    result, result.get("annotated_frame")
                )
                total_unique_events += persist_result.get("created_events", 0)

            # Track unique persons
            for person in result.get("persons", []):
                if person.get("person_id"):
                    unique_persons.add(person["person_id"])

            # Update progress
            progress = int((processed_count * frame_skip / total_frames) * 100)
            processing_manager.update_job(
                job_id,
                processed_frames=processed_count,
                progress=min(progress, 100),
                violations_count=total_unique_events,  # Now shows unique events
                unique_events=total_unique_events,
                persons_count=len(unique_persons),
            )

            # Small delay to prevent blocking
            await asyncio.sleep(0.001)

        # Finalize: close any remaining open violations
        async with async_session() as session:
            persistence = PersistenceManager(session)
            closed_count = await persistence.finalize_video_processing(video_path)

        # Close video writer and finalize the MP4 for browser playback.
        video_writer.release()
        output_path = make_browser_playable_mp4(output_path)

        processing_manager.update_job(
            job_id,
            status="completed",
            progress=100,
            violations_count=total_unique_events,
            unique_events=total_unique_events,
            closed_events=closed_count,
            output_video=str(output_path) if output_path.exists() else None,
        )

    except Exception as e:
        video_writer.release()
        import traceback
        # Get full traceback for better error reporting
        error_trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        # Convert numpy types to strings for JSON serialization
        error_msg = str(e)
        # Limit error message length to prevent issues
        if len(error_trace) > 1000:
            error_trace = error_trace[:1000] + "... (truncated)"
        processing_manager.update_job(job_id, status="failed", error=error_trace)


@router.get("/jobs")
async def list_processing_jobs():
    """List all processing jobs."""
    return {"jobs": processing_manager.list_jobs()}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific processing job."""
    job = processing_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.get("/videos")
async def list_uploaded_videos():
    """List uploaded and validation videos."""
    settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    videos = []
    for path in settings.VIDEOS_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        relative_path = path.relative_to(settings.VIDEOS_DIR)
        parts = relative_path.parts
        source = "uploaded" if len(parts) == 1 else parts[0]
        videos.append(
            {
                "filename": path.name,
                "display_name": str(relative_path),
                "relative_path": str(relative_path),
                "path": str(path),
                "source": source,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
        )

    videos.sort(key=lambda item: (item["source"], item["display_name"].lower()))

    return {"videos": videos}


@router.get("/videos/raw")
async def get_raw_video(video_path: str):
    """Serve a raw uploaded/validation video for browser preview."""
    path = _resolve_video_path(video_path)
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    return FileResponse(
        path,
        media_type=media_types.get(path.suffix.lower(), "application/octet-stream"),
        filename=path.name,
    )


@router.delete("/videos/{filename}")
async def delete_video(filename: str):
    """Delete an uploaded video."""
    video_path = settings.VIDEOS_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {filename}")

    video_path.unlink()
    return {"message": f"Deleted {filename}"}


@router.get("/status")
async def get_processing_status():
    """Get overall processing status."""
    active_jobs = [
        j
        for j in processing_manager.list_jobs()
        if j["status"] in processing_manager.ACTIVE_STATUSES
    ]
    return {
        "active_jobs": len(active_jobs),
        "total_jobs": len(processing_manager.jobs),
    }


@router.get("/processed/{job_id}")
async def get_processed_video(job_id: str):
    """
    Get the annotated/processed video for a completed job.

    Returns the video file with bounding boxes and detection overlays.
    """
    job = processing_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )

    output_video = job.get("output_video")
    if not output_video or not Path(output_video).exists():
        raise HTTPException(
            status_code=404,
            detail="Processed video not found. It may have been deleted."
        )

    return FileResponse(
        output_video,
        media_type="video/mp4",
        filename=Path(output_video).name
    )


@router.get("/processed/{job_id}/stream")
async def stream_processed_video(job_id: str):
    """
    Stream the annotated video as MJPEG for real-time display.

    This is useful for displaying the processed video frame-by-frame in the browser.
    """
    job = processing_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )

    output_video = job.get("output_video")
    if not output_video or not Path(output_video).exists():
        raise HTTPException(
            status_code=404,
            detail="Processed video not found."
        )

    async def generate_frames():
        """Generate MJPEG frames from the processed video."""
        cap = cv2.VideoCapture(output_video)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Encode frame as JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = jpeg.tobytes()

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )

                # Control playback speed (roughly 10 FPS)
                await asyncio.sleep(0.1)
        finally:
            cap.release()

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/processed/list")
async def list_processed_videos():
    """List all processed/annotated videos."""
    processed_dir = settings.PROCESSED_DIR
    if not processed_dir.exists():
        return {"videos": []}

    videos = []
    for path in processed_dir.glob("*.mp4"):
        videos.append({
            "filename": path.name,
            "path": str(path),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        })

    return {"videos": videos}


@router.delete("/processed/{job_id}")
async def delete_processed_video(job_id: str):
    """Delete a processed video."""
    job = processing_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    output_video = job.get("output_video")
    if output_video and Path(output_video).exists():
        Path(output_video).unlink()
        processing_manager.update_job(job_id, output_video=None)
        return {"message": f"Deleted processed video for job {job_id}"}

    return {"message": "No processed video to delete"}


@router.get("/debug/{job_id}")
async def get_debug_info(job_id: str):
    """
    Get debug information for a processing job.
    
    Returns model status, configuration, and detection statistics.
    """
    job = processing_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    pipeline = get_pipeline()
    pipeline.initialize()
    
    # Get detector info
    detector = pipeline.detector
    detector_info = {
        "type": type(detector).__name__,
        "initialized": detector._initialized if hasattr(detector, "_initialized") else False,
    }
    
    # Get YOLOv11 detector info if available
    yolov11_info = {}
    if hasattr(detector, "ppe_detector"):
        ppe_detector = detector.ppe_detector
        if ppe_detector:
            yolov11_info = {
                "model_type": ppe_detector.model_type,
                "model_loaded": ppe_detector.model is not None,
                "device": ppe_detector.device,
                "confidence_threshold": ppe_detector.confidence_threshold,
                "violation_threshold": getattr(ppe_detector, "violation_threshold", None),
                "model_path": str(settings.YOLOV11_MODEL_PATH) if settings.YOLOV11_MODEL_PATH else None,
            }
    
    # Get configuration
    config_info = {
        "detection_confidence_threshold": settings.DETECTION_CONFIDENCE_THRESHOLD,
        "violation_confidence_threshold": getattr(settings, "VIOLATION_CONFIDENCE_THRESHOLD", 0.3),
        "containment_threshold": getattr(settings, "MASK_CONTAINMENT_THRESHOLD", 0.3),
        "temporal_min_frames": getattr(settings, "TEMPORAL_VIOLATION_MIN_FRAMES", 2),
        "required_ppe": settings.REQUIRED_PPE,
    }
    
    # Get job statistics
    job_stats = {
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "processed_frames": job.get("processed_frames", 0),
        "total_frames": job.get("total_frames", 0),
        "violations_count": job.get("violations_count", 0),
        "persons_count": job.get("persons_count", 0),
    }
    
    return {
        "job_id": job_id,
        "job_stats": job_stats,
        "detector_info": detector_info,
        "yolov11_info": yolov11_info,
        "config": config_info,
        "pipeline_stats": pipeline.get_stats(),
    }


@router.get("/live/feed")
async def live_webcam_feed():
    """
    MJPEG stream from webcam with real-time detection annotations.
    
    This endpoint provides a live video feed from the default webcam (index 0)
    with real-time PPE detection and violation overlays. The stream uses MJPEG
    format for browser compatibility.
    
    Performance optimizations:
    - Async processing: Decouples frame capture from ML processing
    - Configurable FPS: Display at high FPS, process at low FPS
    - Result caching: Shows cached detections on skipped frames
    - Smart frame dropping: Prevents queue buildup
    
    Returns:
        StreamingResponse with MJPEG video stream
    """
    from ...ml.stream_processor import generate_live_stream
    
    # Initialize pipeline
    pipeline = get_pipeline()
    pipeline.initialize()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        # Send error frame
        async def error_frame_generator():
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                error_frame,
                "Webcam not available",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            _, buffer = cv2.imencode('.jpg', error_frame)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )
        
        return StreamingResponse(
            error_frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    # Use async stream processor with configured settings
    stream_gen = generate_live_stream(
        cap=cap,
        pipeline=pipeline,
        display_fps=settings.LIVE_STREAM_DISPLAY_FPS,
        process_fps=settings.LIVE_STREAM_PROCESS_FPS,
        queue_size=settings.LIVE_STREAM_QUEUE_SIZE,
        interpolate=settings.LIVE_STREAM_INTERPOLATE,
    )
    
    return StreamingResponse(
        stream_gen,
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
