"use client";

import { useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, ChangeEvent } from 'react';
import styles from './MediaPlayer.module.css';

type MediaKind = 'video' | 'image';

interface MediaPlayerProps {
  src: string;
  kind: MediaKind;
  label: string;
  alt?: string;
  isLive?: boolean;
  isProcessed?: boolean;
  autoPlay?: boolean;
  muted?: boolean;
  errorText?: string;
  onError?: () => void;
}

function formatTime(value: number) {
  if (!Number.isFinite(value) || value <= 0) return '0:00';

  const totalSeconds = Math.floor(value);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = String(totalSeconds % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 5v14l11-7z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 5h4v14H7zM13 5h4v14h-4z" />
    </svg>
  );
}

function VolumeIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 9v6h4l5 4V5L8 9H4z" />
      <path d="M16 8.5a5 5 0 0 1 0 7" />
      <path d="M18.5 6a8 8 0 0 1 0 12" />
    </svg>
  );
}

function MutedIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 9v6h4l5 4V5L8 9H4z" />
      <path d="M18 9l-5 5" />
      <path d="M13 9l5 5" />
    </svg>
  );
}

function FullscreenIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 4H4v4" />
      <path d="M16 4h4v4" />
      <path d="M20 16v4h-4" />
      <path d="M4 16v4h4" />
    </svg>
  );
}

export default function MediaPlayer({
  src,
  kind,
  label,
  alt = 'Video feed',
  isLive = false,
  isProcessed = false,
  autoPlay = false,
  muted = true,
  errorText = 'Media unavailable',
  onError,
}: MediaPlayerProps) {
  const shellRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [isMuted, setIsMuted] = useState(muted);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [hasEnded, setHasEnded] = useState(false);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    setHasError(false);
    setHasEnded(false);
    setCurrentTime(0);
    setDuration(0);
    setIsPlaying(autoPlay);
  }, [autoPlay, kind, src]);

  useEffect(() => {
    setIsMuted(muted);
    if (videoRef.current) videoRef.current.muted = muted;
  }, [muted]);

  const seekPercent = useMemo(() => {
    if (!duration) return 0;
    return Math.max(0, Math.min(100, (currentTime / duration) * 100));
  }, [currentTime, duration]);

  const handleError = () => {
    setHasError(true);
    setIsPlaying(false);
    onError?.();
  };

  const playFromStart = async () => {
    const video = videoRef.current;
    if (!video) return;

    video.currentTime = 0;
    setCurrentTime(0);
    setHasEnded(false);

    try {
      await video.play();
    } catch {
      setIsPlaying(false);
    }
  };

  const togglePlay = async () => {
    const video = videoRef.current;
    if (!video) return;

    if (video.ended) {
      await playFromStart();
      return;
    }

    if (video.paused) {
      try {
        setHasEnded(false);
        await video.play();
      } catch {
        setIsPlaying(false);
      }
      return;
    }

    video.pause();
  };

  const handleSeek = (event: ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    const nextTime = Number(event.target.value);
    if (!video || Number.isNaN(nextTime)) return;

    video.currentTime = nextTime;
    setCurrentTime(nextTime);
    setHasEnded(false);
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;

    const nextMuted = !isMuted;
    video.muted = nextMuted;
    setIsMuted(nextMuted);
  };

  const toggleFullscreen = () => {
    const shell = shellRef.current;
    if (!shell) return;

    if (document.fullscreenElement) {
      void document.exitFullscreen();
      return;
    }

    void shell.requestFullscreen?.();
  };

  const showOffline = !src || hasError;
  const labelClassName = `${styles.badge} ${isLive ? styles.liveBadge : ''} ${
    isProcessed ? styles.processedBadge : ''
  }`;
  const seekStyle = { '--seek-progress': `${seekPercent}%` } as CSSProperties;

  return (
    <div className={styles.player} ref={shellRef}>
      <div className={styles.frame}>
        {showOffline ? (
          <div className={styles.offline} role="status">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M2 2l20 20" />
              <path d="M15 15a4 4 0 0 1-5.8-5.8" />
              <path d="M12 12v.01" />
              <path d="M10 5.4a4 4 0 0 1 5.6 5.6" />
              <path d="M8 8a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-4a2 2 0 0 0-2-2" />
            </svg>
            <span>{errorText}</span>
          </div>
        ) : kind === 'image' ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={src} alt={alt} className={styles.media} onError={handleError} />
        ) : (
          <video
            ref={videoRef}
            src={src}
            className={styles.media}
            muted={isMuted}
            playsInline
            autoPlay={autoPlay}
            preload="metadata"
            onLoadedMetadata={(event) => {
              setDuration(event.currentTarget.duration || 0);
            }}
            onDurationChange={(event) => {
              setDuration(event.currentTarget.duration || 0);
            }}
            onTimeUpdate={(event) => {
              setCurrentTime(event.currentTarget.currentTime || 0);
            }}
            onPlay={() => {
              setIsPlaying(true);
              setHasEnded(false);
            }}
            onPause={() => setIsPlaying(false)}
            onEnded={() => {
              setIsPlaying(false);
              setHasEnded(true);
            }}
            onError={handleError}
          />
        )}

        <div className={labelClassName}>
          {isLive ? <span className={styles.liveDot} /> : null}
          {label}
        </div>

        {isProcessed && hasEnded ? (
          <div className={styles.replayLayer}>
            <div className={styles.replayCard}>
              <span>Output Complete</span>
              <strong>Replay analyzed video?</strong>
              <button type="button" onClick={playFromStart}>
                Replay Output
              </button>
            </div>
          </div>
        ) : null}
      </div>

      {kind === 'video' ? (
        <div className={styles.controls}>
          <button
            type="button"
            className={styles.iconButton}
            onClick={togglePlay}
            disabled={showOffline}
            aria-label={isPlaying ? 'Pause video' : 'Play video'}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <PauseIcon /> : <PlayIcon />}
          </button>

          <span className={styles.time}>
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>

          <input
            className={styles.seek}
            type="range"
            min="0"
            max={duration || 0}
            step="0.1"
            value={Math.min(currentTime, duration || 0)}
            style={seekStyle}
            disabled={showOffline || !duration}
            aria-label="Seek video"
            onChange={handleSeek}
          />

          <button
            type="button"
            className={styles.iconButton}
            onClick={toggleMute}
            disabled={showOffline}
            aria-label={isMuted ? 'Unmute video' : 'Mute video'}
            title={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted ? <MutedIcon /> : <VolumeIcon />}
          </button>

          <button
            type="button"
            className={styles.iconButton}
            onClick={toggleFullscreen}
            disabled={showOffline}
            aria-label="Fullscreen"
            title="Fullscreen"
          >
            <FullscreenIcon />
          </button>
        </div>
      ) : null}
    </div>
  );
}
