"""Low-bandwidth webcam/microphone capture and media decoding helpers."""
import io
import os
import subprocess
import tempfile
import threading
from collections import deque
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

class LiveCaptureBuffer:
    """Thread-safe per-session buffer for sampled webcam frames and microphone audio."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = deque(maxlen=8)
        self.audio_chunks = []
        self.audio_samples = 0
        self.video_counter = 0
        self.sample_rate = 48000

    def clear(self):
        with self.lock:
            self.frames.clear()
            self.audio_chunks.clear()
            self.audio_samples = 0
            self.video_counter = 0
            self.sample_rate = 48000

    def add_video(self, frame):
        # Keep the network stream light: downscale before encoding and sample
        # only about once per second instead of processing every incoming frame.
        height, width = frame.shape[:2]
        if width > 480:
            target_h = max(1, int(height * (480 / width)))
            frame = cv2.resize(frame, (480, target_h), interpolation=cv2.INTER_AREA)
        with self.lock:
            self.video_counter += 1
            if self.video_counter % 10 == 0 and len(self.frames) < 8:
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok:
                    self.frames.append(encoded.tobytes())

    def add_audio(self, frame):
        data = frame.to_ndarray()
        if data.size == 0:
            return
        if data.ndim == 2:
            data = data.mean(axis=0)
        data = np.asarray(data).reshape(-1)
        with self.lock:
            self.sample_rate = int(frame.sample_rate or self.sample_rate)
            remaining = max(0, int(self.sample_rate * 30) - self.audio_samples)
            if remaining <= 0:
                return
            data = data[:remaining]
            if np.issubdtype(data.dtype, np.integer):
                data = np.clip(data, -32768, 32767).astype(np.int16)
            else:
                data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
            self.audio_chunks.append(data.copy())
            self.audio_samples += len(data)

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


class MicrophoneAudioProcessor:
    def __init__(self, buffer):
        self.buffer = buffer

    def recv(self, frame):
        self.buffer.add_audio(frame)
        return frame


def extract_video_frames(video_bytes, max_frames=5):
    """Sample a few frames from an uploaded video."""
    temp_path = None
    try:
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video.write(video_bytes)
            temp_path = temp_video.name
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return [], ["Could not open this video. Try an MP4 or MOV file."]
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return [], ["No readable frames were found in the video."]
        indexes = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
        frames = []
        for index in sorted(set(int(i) for i in indexes)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                encoded_ok, encoded = cv2.imencode(".jpg", frame)
                if encoded_ok:
                    frames.append(encoded.tobytes())
        cap.release()
        return frames, [f"Sampled {len(frames)} frame(s) across the uploaded video."]
    except Exception as exc:
        return [], [f"Video frame extraction failed ({type(exc).__name__})."]
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def extract_video_audio(video_bytes):
    """Extract a mono WAV audio track from a video using bundled FFmpeg."""
    video_path = audio_path = None
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_file.write(video_bytes)
            video_path = video_file.name
        audio_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = audio_handle.name
        audio_handle.close()
        result = subprocess.run(
            [ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", audio_path],
            capture_output=True, text=True, timeout=90
        )
        if result.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) < 44:
            return None, ["No readable audio track was found in the video."]
        with open(audio_path, "rb") as audio_file:
            return audio_file.read(), ["Extracted the video's audio track for acoustic features and speech transcription."]
    except subprocess.TimeoutExpired:
        return None, ["Audio extraction timed out. Try a shorter video."]
    except Exception as exc:
        return None, [f"Could not extract video audio ({type(exc).__name__})."]
    finally:
        for path in (video_path, audio_path):
            if path and os.path.exists(path):
                os.remove(path)
