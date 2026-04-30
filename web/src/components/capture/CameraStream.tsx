"use client";

import { useEffect, useRef, useState } from "react";

import { useCaptureStore } from "@/lib/captureSession";

interface Props {
  /** Active when the WS is open + the user has tapped Start. The
   *  component still renders the video element when `false` so the
   *  camera preview is visible while the user picks settings, but
   *  no frames are pushed to the server. */
  capturing: boolean;
  /** Frames per second to capture and send. 10 Hz is plenty for
   *  SLAM tracking on a phone-camera feed. */
  fps?: number;
  /** `enumerateDevices()` lets the user pick when there are multiple
   *  cameras (desktop with both built-in + USB, phone with front +
   *  rear). When unset, we default to `facingMode: "environment"`
   *  (rear-facing) for the mobile case. */
  deviceId?: string;
}

/**
 * Wraps `navigator.mediaDevices.getUserMedia` and emits a JPEG frame
 * every `1 / fps` seconds while `capturing` is true. The frames go
 * through the captureSession store's `sendFrame()` action, which
 * handles the WebSocket binary write + client-side backpressure.
 *
 * The video element is the only DOM that bubbles back up to the
 * caller — the page lays it out as the background of the capture UI
 * (full-screen on portrait, left half on landscape). The frame-grab
 * happens via an OffscreenCanvas the user never sees.
 */
export function CameraStream({ capturing, fps = 10, deviceId }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sendFrame = useCaptureStore((s) => s.sendFrame);
  const setVideoState = useCaptureStore((s) => s.setVideoState);

  // Open / close the camera. We also re-acquire when `deviceId`
  // changes so the camera-source picker takes effect immediately.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    async function acquire() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError(
          "this browser doesn't expose getUserMedia — load over HTTPS, " +
            "or use the upload flow instead.",
        );
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: deviceId
            ? { deviceId: { exact: deviceId } }
            : {
                // Rear camera on phones; ignored on desktops.
                facingMode: { ideal: "environment" },
                // Cap resolution: above ~720p the SLAM step takes
                // longer than the frame interval and we backpressure
                // anyway. 720p is a good sweet spot.
                width: { ideal: 1280 },
                height: { ideal: 720 },
              },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        const v = videoRef.current;
        if (v) {
          v.srcObject = stream;
          // iOS Safari: muted + playsInline + manual play() is the
          // only combo that consistently autostarts the preview.
          v.muted = true;
          v.playsInline = true;
          await v.play().catch(() => undefined);
        }
      } catch (e) {
        if (!cancelled) setError(String((e as Error).message));
      }
    }
    void acquire();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      const v = videoRef.current;
      if (v) v.srcObject = null;
    };
  }, [deviceId]);

  // Frame-grab loop. Runs only while `capturing` is true. Uses an
  // OffscreenCanvas (or a hidden 2D canvas for browsers without
  // OffscreenCanvas, which still includes some Safari builds) to
  // draw the current video frame, encode as JPEG, hand to the
  // store. setInterval is fine; precise timing isn't required and
  // it lets us skip whole intervals if the page goes background.
  useEffect(() => {
    if (!capturing) return;
    const v = videoRef.current;
    if (!v) return;
    const intervalMs = Math.max(50, Math.round(1000 / fps));
    let canvas: HTMLCanvasElement | null = null;
    let ctx: CanvasRenderingContext2D | null = null;

    const id = window.setInterval(() => {
      const ready = v.readyState >= 2 && v.videoWidth > 0;
      // Push readiness + resolution into the store every tick so the
      // capture page chip can show "video: 1280×720 · ws: open · sent
      // N · processed M". A stuck capture has one of these stuck at
      // 0 / false; surfacing all three makes the failure mode obvious
      // from the phone instead of needing to ssh into the studio.
      setVideoState(
        ready,
        ready ? [v.videoWidth, v.videoHeight] : null,
      );
      if (!ready) return;
      if (!canvas) {
        canvas = document.createElement("canvas");
        canvas.width = v.videoWidth;
        canvas.height = v.videoHeight;
        ctx = canvas.getContext("2d");
      }
      if (!ctx) return;
      // Resize if the camera switched resolutions on us.
      if (
        canvas.width !== v.videoWidth ||
        canvas.height !== v.videoHeight
      ) {
        canvas.width = v.videoWidth;
        canvas.height = v.videoHeight;
      }
      ctx.drawImage(v, 0, 0);
      canvas.toBlob(
        (blob) => {
          if (blob) sendFrame(blob);
        },
        "image/jpeg",
        0.7,
      );
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [capturing, fps, sendFrame, setVideoState]);

  return (
    <>
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          background: "#000",
        }}
      />
      {error && (
        <div
          role="status"
          style={{
            position: "absolute",
            inset: 0,
            display: "grid",
            placeItems: "center",
            background: "rgba(10, 10, 10, 0.85)",
            color: "#fff",
            fontSize: "var(--fs-sm)",
            padding: 16,
            textAlign: "center",
          }}
        >
          {error}
        </div>
      )}
    </>
  );
}
