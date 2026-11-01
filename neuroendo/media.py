"""Low-bandwidth webcam/microphone capture, held only in memory.

Privacy: nothing is ever written to disk. Frames and audio live in a per-session
buffer until the recording is analysed, then the buffer is wiped (see wipe()).
"""
import io
import threading
from collections import deque
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

class LiveCaptureBuffer:
    """Thread-safe per-session buffer for sampled webcam frames and microphone audio."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = []
        self.audio_chunks = []
        self.audio_samples = 0
        self.video_counter = 0
        self.stride = 10          # keep every Nth incoming frame
        self.sample_rate = 48000

    MAX_FRAMES = 8

    def clear(self):
        """Delete every captured frame and audio sample from memory."""
        with self.lock:
            for chunk in self.audio_chunks:
                chunk.fill(0)             # overwrite raw audio before dropping it
            self.frames = []
            self.audio_chunks = []
            self.audio_samples = 0
            self.video_counter = 0
            self.stride = 10
            self.sample_rate = 48000

    wipe = clear

    def is_empty(self):
        with self.lock:
            return not self.frames and self.audio_samples == 0

    def add_video(self, frame):
        # Keep the network stream light: downscale before encoding and sample
        # only about once per second instead of processing every incoming frame.
        height, width = frame.shape[:2]
        if width > 480:
            target_h = max(1, int(height * (480 / width)))
            frame = cv2.resize(frame, (480, target_h), interpolation=cv2.INTER_AREA)
        with self.lock:
            self.video_counter += 1
            if self.video_counter % self.stride:
                return
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                return
            self.frames.append(encoded.tobytes())
            # Spread the 8 kept frames over the whole recording: when full, drop
            # every other frame and sample half as often from now on.
            if len(self.frames) >= self.MAX_FRAMES:
                self.frames = self.frames[::2]
                self.stride *= 2

    def add_audio(self, frame):
        mono = audio_frame_to_mono(frame)
        if mono.size == 0:
            return
        with self.lock:
            self.sample_rate = int(frame.sample_rate or self.sample_rate)
            remaining = max(0, int(self.sample_rate * 30) - self.audio_samples)
            if remaining <= 0:
                return
            data = (np.clip(mono[:remaining], -1.0, 1.0) * 32767).astype(np.int16)
            self.audio_chunks.append(data)
            self.audio_samples += len(data)

    def has_audio(self):
        with self.lock:
            return self.audio_samples > 0

    def snapshot(self):
        with self.lock:
            frames = list(self.frames)
            chunks = [chunk.copy() for chunk in self.audio_chunks]
            rate = int(self.sample_rate)
        audio_bytes = None
        if chunks:
            from scipy.io import wavfile
            buffer = io.BytesIO()
            wavfile.write(buffer, rate, np.concatenate(chunks).astype(np.int16))
            audio_bytes = buffer.getvalue()
        return frames, audio_bytes


class WebcamVideoProcessor:
    def __init__(self, buffer):
        self.buffer = buffer

    def recv(self, frame):
        self.buffer.add_video(frame.to_ndarray(format="bgr24"))
        return frame


def audio_frame_to_mono(frame):
    """Convert a WebRTC audio frame to mono float32 in [-1, 1].

    Browsers usually send packed (interleaved) stereo s16: to_ndarray() returns
    shape (1, samples*channels) as L R L R ... It must be de-interleaved before
    averaging, otherwise the clip is twice as long and scrambled. Planar formats
    return (channels, samples) instead.
    """
    raw = frame.to_ndarray()
    if raw.size == 0:
        return np.zeros(0, dtype=np.float32)
    channels = max(1, len(frame.layout.channels))
    if frame.format.is_planar:
        data = raw.reshape(channels, -1).astype(np.float32).mean(axis=0)
    else:
        data = raw.reshape(-1, channels).astype(np.float32).mean(axis=1)
    if np.issubdtype(raw.dtype, np.integer):
        data /= float(np.iinfo(raw.dtype).max)
    return data


class MicrophoneAudioProcessor:
    def __init__(self, buffer):
        self.buffer = buffer

    def recv(self, frame):
        self.buffer.add_audio(frame)
        # Send silence back so the browser doesn't play the user's own voice (echo).
        import av
        silent = av.AudioFrame.from_ndarray(np.zeros_like(frame.to_ndarray()),
                                            format=frame.format.name, layout=frame.layout.name)
        silent.sample_rate = frame.sample_rate
        silent.pts, silent.time_base = frame.pts, frame.time_base
        return silent
