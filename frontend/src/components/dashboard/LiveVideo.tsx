"use client";

import { useEffect, useMemo, useState } from 'react';
import {
  fetchProcessingJob,
  fetchVideos,
  processedVideoStreamUrl,
  rawVideoUrl,
  startVideoProcessing,
  VIDEO_STREAM_URL,
  ProcessingJob,
  VideoAsset,
} from '@/lib/api';
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
  const folder = video.relative_path.split(/[\\/]/).slice(-2, -1)[0];
  const cleanFilename = video.filename.replace(/_/g, ' ');
  return folder ? `${cleanFilename} (${folder})` : cleanFilename;
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

  useEffect(() => {
    let active = true;

    async function loadVideos() {
      try {
        const data = await fetchVideos();
        if (!active) return;

        const ppeVideos = data.videos.filter(isPpeAnalysisVideo);
        const selectableVideos = ppeVideos.length > 0 ? ppeVideos : data.videos;

        setVideos(selectableVideos);
        const preferred =
          selectableVideos.find((video) => video.filename.toLowerCase().includes('part1')) ??
          selectableVideos.find((video) => video.display_name.toLowerCase().includes('ppe_video')) ??
          selectableVideos.find((video) => video.display_name.toLowerCase().includes('ppe_red_zone')) ??
          selectableVideos[0];
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

  const selectedVideo = useMemo(
    () => videos.find((video) => video.path === selectedPath) ?? null,
    [selectedPath, videos],
  );

  const isProcessing = job?.status === 'pending' || job?.status === 'processing';
  const isRunDisabled =
    mode !== 'validation' || loadingVideos || !selectedVideo || isProcessing || starting;
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
  };

  const videoSrc =
    mode === 'live'
      ? VIDEO_STREAM_URL
      : job?.status === 'completed' && job.output_video
        ? processedVideoStreamUrl(job.id)
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
              <label className={styles.fieldGroup}>
                <span>PPE validation video</span>
                <select
                  value={selectedPath}
                  disabled={loadingVideos || videos.length === 0 || isProcessing || starting}
                  onChange={(event) => {
                    setSelectedPath(event.target.value);
                    setJob(null);
                    setActionError('');
                    setNotice('');
                    setHasError(false);
                  }}
                >
                  {videos.length === 0 ? (
                    <option value="">No videos available</option>
                  ) : (
                    videos.map((video) => (
                      <option key={video.path} value={video.path}>
                        {formatVideoLabel(video)} - {video.size_mb} MB
                      </option>
                    ))
                  )}
                </select>
              </label>

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

      <div className={styles.videoStage}>
        {!hasError && videoSrc ? (
          mode === 'validation' && !(job?.status === 'completed' && job.output_video) ? (
            <video
              key={videoSrc}
              src={videoSrc}
              className={styles.videoFeed}
              controls
              muted
              playsInline
              onError={() => setHasError(true)}
            />
          ) : (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                key={videoSrc}
                src={videoSrc}
                alt={mode === 'live' ? 'Live camera feed' : 'Processed validation feed'}
                className={styles.videoFeed}
                onError={() => setHasError(true)}
              />
            </>
          )
        ) : (
          <div className={styles.offlineMessage}>
            <svg className={styles.offlineIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 2l20 20"></path><path d="M15 15a4 4 0 01-5.8-5.8"></path><path d="M12 12v.01"></path><path d="M10 5.4a4 4 0 015.6 5.6"></path><path d="M8 8a2 2 0 00-2 2v4a2 2 0 002 2h8a2 2 0 002-2v-4a2 2 0 00-2-2"></path>
            </svg>
            <p>{mode === 'live' ? 'Camera feed offline' : 'Video unavailable'}</p>
          </div>
        )}

        <div className={styles.overlay}>
          {mode === 'live' ? <div className={styles.recDot}></div> : null}
          {mode === 'live' ? 'LIVE' : job?.status === 'completed' && job.output_video ? 'PROCESSED' : 'DATASET'}
        </div>
      </div>
    </div>
  );
}
