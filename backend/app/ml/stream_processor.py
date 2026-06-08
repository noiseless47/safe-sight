"""
Async Stream Processor for Live Video Feeds

Decouples frame capture from ML processing to enable smooth streaming
at high FPS while processing at a lower rate.
"""

import asyncio
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Lock

from ..core.database import async_session
from ..services.persistence import PersistenceManager

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Asynchronous stream processor for live video feeds.
    
    Separates frame capture (high FPS) from ML processing (low FPS)
    by using a processing queue and result caching.
    """
    
    def __init__(
        self,
        pipeline,
        display_fps: int = 30,
        process_fps: int = 3,
        queue_size: int = 2,
        interpolate: bool = False
    ):
        """
        Args:
            pipeline: ML detection pipeline instance
            display_fps: Target display frame rate
            process_fps: ML processing frame rate (lower = less lag)
            queue_size: Max frames in processing queue
            interpolate: Whether to interpolate bounding boxes
        """
        self.pipeline = pipeline
        self.display_fps = display_fps
        self.process_fps = process_fps
        self.queue_size = queue_size
        self.interpolate = interpolate
        
        # Processing queue and results cache
        self.frame_queue = Queue(maxsize=queue_size)
        self.result_cache: Optional[Dict[str, Any]] = None
        self.result_lock = Lock()
        
        # Worker thread
        self.worker_thread: Optional[Thread] = None
        self.running = False
        
        logger.info(
            f"StreamProcessor initialized: "
            f"display={display_fps}fps, process={process_fps}fps, "
            f"queue_size={queue_size}, interpolate={interpolate}"
        )
    
    def start(self):
        """Start processing worker thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()
        logger.info("StreamProcessor worker thread started")
    
    def stop(self):
        """Stop processing worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        logger.info("StreamProcessor worker thread stopped")
    
    def _processing_worker(self):
        """Background worker that processes frames from the queue."""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self.running:
                try:
                    # Get frame from queue (non-blocking with timeout)
                    frame, video_source = self.frame_queue.get(timeout=0.5)
                    
                    # Process through ML pipeline
                    result = self.pipeline.process_frame(frame, video_source=video_source)
                    
                    # Add video_source to result for persistence
                    result["video_source"] = video_source
                    
                    # Persist to database asynchronously
                    try:
                        loop.run_until_complete(self._persist_result(result, frame))
                    except Exception as db_error:
                        logger.error(f"Database persistence error: {db_error}", exc_info=True)
                    
                    # Update result cache
                    with self.result_lock:
                        self.result_cache = result
                    
                    self.frame_queue.task_done()
                    
                except Empty:
                    # No frame to process, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error in processing worker: {e}", exc_info=True)
        finally:
            loop.close()
    
    async def _persist_result(self, result: Dict[str, Any], frame: np.ndarray):
        """
        Persist detection results to the database.
        
        Args:
            result: Detection result from ML pipeline
            frame: Original frame (for annotated frame storage)
        """
        async with async_session() as session:
            persistence = PersistenceManager(session)
            annotated_frame = result.get("annotated_frame")
            await persistence.persist_frame_results(result, annotated_frame)
    
    def submit_frame(self, frame: np.ndarray, video_source: str = "webcam"):
        """
        Submit a frame for processing.
        
        If the queue is full, drops the oldest frame to avoid blocking.
        
        Args:
            frame: Video frame to process
            video_source: Source identifier
        """
        try:
            # Try to add frame to queue (non-blocking)
            self.frame_queue.put_nowait((frame, video_source))
        except:
            # Queue is full - drop oldest frame and add new one
            try:
                self.frame_queue.get_nowait()  # Remove oldest
                self.frame_queue.put_nowait((frame, video_source))
                logger.debug("Frame queue full, dropped oldest frame")
            except:
                pass
    
    def get_latest_result(self, fallback_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get the latest processing result.
        
        Args:
            fallback_frame: Frame to use if no result is cached yet
            
        Returns:
            Latest detection result with annotated frame
        """
        with self.result_lock:
            if self.result_cache is not None:
                return self.result_cache
            else:
                # No result yet - return empty result with fallback frame
                return {
                    "persons": [],
                    "annotated_frame": fallback_frame,
                    "frame_number": 0,
                    "timestamp": datetime.now().isoformat(),
                }


async def generate_live_stream(
    cap: cv2.VideoCapture,
    pipeline,
    display_fps: int = 30,
    process_fps: int = 3,
    queue_size: int = 2,
    interpolate: bool = False,
):
    """
    Async generator for live video stream with ML processing.
    
    Yields JPEG-encoded frames for MJPEG streaming.
    
    Args:
        cap: OpenCV VideoCapture instance
        pipeline: ML detection pipeline
        display_fps: Target display frame rate
        process_fps: ML processing frame rate
        queue_size: Max frames in processing queue
        interpolate: Whether to interpolate bounding boxes
        
    Yields:
        JPEG-encoded frame bytes in multipart format
    """
    processor = StreamProcessor(
        pipeline=pipeline,
        display_fps=display_fps,
        process_fps=process_fps,
        queue_size=queue_size,
        interpolate=interpolate
    )
    
    # Start background processing
    processor.start()
    
    try:
        # Calculate frame skip for processing
        # Process every Nth frame based on fps ratio
        if process_fps >= display_fps:
            process_skip = 1  # Process every frame
        else:
            process_skip = max(1, int(display_fps / process_fps))
        
        frame_count = 0
        display_interval = 1.0 / display_fps  # Time between frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Submit frame for processing at reduced rate
            if frame_count % process_skip == 0:
                processor.submit_frame(frame, video_source="webcam")
            
            # Always get latest result for display
            result = processor.get_latest_result(fallback_frame=frame)
            annotated = result.get("annotated_frame", frame)
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
            
            # Control display frame rate
            await asyncio.sleep(display_interval)
            
    except Exception as e:
        logger.error(f"Error in live stream generator: {e}", exc_info=True)
    finally:
        # Cleanup
        processor.stop()
        logger.info("Live stream generator stopped")
