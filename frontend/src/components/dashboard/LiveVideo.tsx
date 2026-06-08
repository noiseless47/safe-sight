"use client";

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchProcessingJob,
  fetchVideos,
  processedVideoUrl,
  rawVideoUrl,
  startVideoProcessing,
  VIDEO_STREAM_URL,
  ProcessingJob,
  VideoAsset,
} from '@/lib/api';
import MediaPlayer from './MediaPlayer';
import styles from './LiveVideo.module.css';

function formatError(error: unknown) {
  if (error instanceof Error && error.message) return error.message;
  return 'Could not start analysis';
}

function progressOf(job: ProcessingJob | null) {
  if (!job) return 0;
  return Math.max(0, Math.min(100, Math.round(job.progress || 0)));
}

function isPpeAnalysisVideo(video: VideoAsset) {
  const searchable = `${video.filename} ${video.display_name} ${video.relative_path}`.toLowerCase();
  return (
    video.source === 'uploaded' ||
    searchable.includes('ppe_video') ||
    searchable.includes('ppe_red_zone') ||
    searchable.includes('hardhat') ||
    searchable.includes('helmet')
  );
}

function formatVideoLabel(video: VideoAsset) {
  const stem = video.filename.replace(/\.[^.]+$/, '').replace(/^\d+[_\-\s]*/, '');
  return stem
    .split(/[_\-\s]+/)
    .filter(Boolean)
    .map((token) => {
      const upperToken = token.toUpperCase();
      if (upperToken === 'PPE' || upperToken === 'CCTV') return upperToken;
      return token.charAt(0).toUpperCase() + token.slice(1);
    })
    .join(' ');
}

function videoPriority(video: VideoAsset) {
  const filename = video.filename.toLowerCase();
  const orderMatch = filename.match(/^(\d+)/);
  if (orderMatch) return Number(orderMatch[1]);
  return 99;
}

export default function LiveVideo() {
  const [mode, setMode] = useState<'live' | 'validation'>('validation');
  const [hasError, setHasError] = useState(false);
  const [videos, setVideos] = useState<VideoAsset[]>([]);
  const [selectedPath, setSelectedPath] = useState('');
  const [job, setJob] = useState<ProcessingJob | null>(null);
  const [loadingVideos, setLoadingVideos] = useState(true);
  const [starting, setStarting] = useState(false);
  const [actionError, setActionError] = useState('');
  const [notice, setNotice] = useState('');
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;

    async function loadVideos() {
      try {
        const data = await fetchVideos();
        if (!active) return;

        const ppeVideos = data.videos.filter(isPpeAnalysisVideo);
        const selectableVideos = [...(ppeVideos.length > 0 ? ppeVideos : data.videos)].sort(
          (first, second) =>
            videoPriority(first) - videoPriority(second) ||
            formatVideoLabel(first).localeCompare(formatVideoLabel(second)),
        );

        setVideos(selectableVideos);
        const preferred = selectableVideos[0];
        if (preferred) setSelectedPath(preferred.path);
      } catch (error) {
        console.error(error);
        if (active) setActionError('Video dataset unavailable. Backend video listing is unreachable.');
      } finally {
        if (active) setLoadingVideos(false);
      }
    }

    loadVideos();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!job || (job.status !== 'pending' && job.status !== 'processing')) return;

    let active = true;
    const pollJob = async () => {
      try {
        const nextJob = await fetchProcessingJob(job.id);
        if (active) setJob(nextJob);
      } catch (error) {
        console.error(error);
        if (active) {
          setJob(null);
          setNotice('');
          setActionError('Analysis session expired. Start it again.');
        }
      }
    };

    const interval = window.setInterval(pollJob, 1500);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (!dropdownOpen) return;

    function handlePointerDown(event: PointerEvent) {
      if (!dropdownRef.current?.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setDropdownOpen(false);
    }

    document.addEventListener('pointerdown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [dropdownOpen]);

  useEffect(() => {
    if (job?.status === 'completed' && job.output_video) {
      setHasError(false);
    }
  }, [job?.id, job?.output_video, job?.status]);

  const selectedVideo = useMemo(
    () => videos.find((video) => video.path === selectedPath) ?? null,
    [selectedPath, videos],
  );

  const isProcessing = job?.status === 'pending' || job?.status === 'processing';
  const isRunDisabled =
    mode !== 'validation' || loadingVideos || !selectedVideo || isProcessing || starting;
  const isDropdownDisabled = loadingVideos || videos.length === 0 || isProcessing || starting;
  const jobProgress = progressOf(job);

  const handleProcess = async () => {
    if (!selectedVideo || isProcessing || starting) return;

    setStarting(true);
    setActionError('');
    setNotice('');
    setHasError(false);

    try {
      const started = await startVideoProcessing(selectedVideo.path);
      const nextJob = await fetchProcessingJob(started.job_id);
      setJob(nextJob);
      setNotice(started.reused ? 'Resumed the active analysis for this video.' : 'Analysis job accepted.');
    } catch (error) {
      console.error(error);
      setActionError(formatError(error));
    } finally {
      setStarting(false);
    }
  };

  const handleModeChange = (nextMode: 'live' | 'validation') => {
    setMode(nextMode);
    setHasError(false);
    setActionError('');
    setNotice('');
    setDropdownOpen(false);
  };

  const handleVideoSelect = (video: VideoAsset) => {
    setSelectedPath(video.path);
    setJob(null);
    setActionError('');
    setNotice('');
    setHasError(false);
    setDropdownOpen(false);
  };

  const isProcessedOutput = mode === 'validation' && job?.status === 'completed' && Boolean(job.output_video);
  const videoSrc =
    mode === 'live'
      ? VIDEO_STREAM_URL
      : isProcessedOutput
        ? processedVideoUrl(job.id)
        : selectedVideo
          ? rawVideoUrl(selectedVideo.path)
          : '';

  const runButtonText = starting
    ? 'Starting'
    : isProcessing
      ? 'Analyzing'
      : job?.status === 'failed'
        ? 'Retry Analysis'
        : job?.status === 'completed'
          ? 'Run Again'
          : 'Run PPE Analysis';

  const statusTitle = actionError
    ? 'Action needed'
    : starting
      ? 'Starting analysis'
      : job?.status === 'failed'
        ? 'Analysis failed'
        : job?.status === 'completed'
          ? 'Processed output ready'
          : isProcessing
            ? 'Analyzing video'
            : 'Ready';

  const statusDetail = actionError
    ? actionError
    : starting
      ? 'Submitting analysis job to backend'
      : job?.status === 'failed'
        ? 'The backend marked this run as failed. Retry after the current process clears.'
        : job?.status === 'completed'
          ? `${job.unique_events ?? job.violations_count ?? 0} unique events across ${job.processed_frames} sampled frames`
          : isProcessing
            ? `${jobProgress}% complete - ${job?.processed_frames ?? 0}/${job?.total_frames ?? 0} frames - ${job?.unique_events ?? job?.violations_count ?? 0} events`
            : selectedVideo
              ? `${videos.length} PPE-ready videos - ${selectedVideo.size_mb} MB selected`
              : loadingVideos
                ? 'Loading validation videos from backend'
                : 'No validation videos found';

  const jobStripTitle = (starting || isProcessing) && notice ? notice : statusTitle;

  return (
    <div className={`glass-panel ${styles.container}`}>
      <div className={styles.analysisPanel}>
        <div className={styles.panelTop}>
          <div className={styles.segmented} aria-label="Video source mode">
            <button
              type="button"
              className={mode === 'validation' ? styles.activeSegment : ''}
              onClick={() => handleModeChange('validation')}
            >
              Dataset
            </button>
            <button
              type="button"
              className={mode === 'live' ? styles.activeSegment : ''}
              onClick={() => handleModeChange('live')}
            >
              Live
            </button>
          </div>

          <div className={`${styles.statePill} ${actionError || job?.status === 'failed' ? styles.failed : isProcessing ? styles.processing : job?.status === 'completed' ? styles.completed : ''}`}>
            <span className={styles.stateDot} />
            {mode === 'live' ? 'Live Camera' : statusTitle}
          </div>
        </div>

        {mode === 'validation' ? (
          <>
            <div className={styles.datasetRow}>
              <div className={styles.fieldGroup}>
                <span className={styles.fieldLabel}>PPE validation video</span>
                <div className={styles.dropdown} ref={dropdownRef}>
                  <button
                    type="button"
                    className={styles.dropdownTrigger}
                    disabled={isDropdownDisabled}
                    aria-haspopup="listbox"
                    aria-expanded={dropdownOpen}
                    onClick={() => setDropdownOpen((open) => !open)}
                  >
                    <span className={styles.dropdownMain}>
                      {selectedVideo ? formatVideoLabel(selectedVideo) : 'No videos available'}
                    </span>
                    <span className={styles.dropdownMeta}>
                      {selectedVideo ? `${selectedVideo.size_mb} MB` : 'Waiting for backend'}
                    </span>
                    <svg viewBox="0 0 20 20" aria-hidden="true" className={styles.chevron}>
                      <path d="M5 7.5 10 12.5 15 7.5" />
                    </svg>
                  </button>

                  {dropdownOpen ? (
                    <div className={styles.dropdownMenu} role="listbox" aria-label="PPE validation videos">
                      {videos.map((video) => {
                        const isSelected = video.path === selectedPath;
                        return (
                          <button
                            key={video.path}
                            type="button"
                            className={`${styles.dropdownOption} ${isSelected ? styles.selectedOption : ''}`}
                            role="option"
                            aria-selected={isSelected}
                            onClick={() => handleVideoSelect(video)}
                          >
                            <span>{formatVideoLabel(video)}</span>
                            <strong>{video.size_mb} MB</strong>
                          </button>
                        );
                      })}
                    </div>
                  ) : null}
                </div>
                {videos.length === 0 && !loadingVideos ? (
                  <span className={styles.fieldHint}>No validation videos found</span>
                ) : null}
              </div>

              <button
                type="button"
                className={styles.runButton}
                onClick={handleProcess}
                disabled={isRunDisabled}
              >
                {starting || isProcessing ? <span className={styles.spinner} /> : null}
                {runButtonText}
              </button>
            </div>

            <div className={`${styles.jobStrip} ${actionError || job?.status === 'failed' ? styles.failedStrip : ''}`}>
              <div className={styles.jobCopy}>
                <span>{jobStripTitle}</span>
                <strong>{statusDetail}</strong>
              </div>
              <div className={styles.progressTrack} aria-hidden="true">
                <div className={styles.progressFill} style={{ width: `${starting ? 8 : jobProgress}%` }} />
              </div>
            </div>
          </>
        ) : (
          <div className={styles.liveHint}>Waiting for the configured camera feed.</div>
        )}
      </div>

      <MediaPlayer
        src={hasError ? '' : videoSrc}
        kind={mode === 'live' ? 'image' : 'video'}
        label={mode === 'live' ? 'LIVE' : isProcessedOutput ? 'PROCESSED' : 'DATASET'}
        alt={mode === 'live' ? 'Live camera feed' : 'PPE validation video'}
        isLive={mode === 'live'}
        isProcessed={isProcessedOutput}
        autoPlay={isProcessedOutput}
        muted
        errorText={mode === 'live' ? 'Camera feed offline' : 'Video unavailable'}
        onError={() => setHasError(true)}
      />
    </div>
  );
}
