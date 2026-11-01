"""Emotion cue extraction and pretrained text, face, and speech inference."""
import io
import re
import numpy as np
import streamlit as st
import logging
import threading
from .learning import clip

log = logging.getLogger("neuroendo")

TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
FACE_MODEL = "trpakov/vit-face-expression"


@st.cache_resource(show_spinner="Loading an AI model (first time only)…")
def _load_pipeline(task, model):
    """Load a pretrained pipeline ONCE per server and share it across sessions.

    Models hold no user data, so sharing them is safe; loading one copy per
    visitor would exhaust memory. A failed load is cached as None so the app
    doesn't retry the download on every frame. Returns (pipeline_or_None, lock).
    """
    try:
        from transformers import pipeline
        kwargs = {"top_k": None} if task == "text-classification" else {}
        return pipeline(task, model=model, **kwargs), threading.Lock()
    except Exception as exc:  # network/model problems: log details server-side only
        log.warning("Could not load %s (%s): %s", model, task, type(exc).__name__)
        return None, threading.Lock()


def _run(task, model, *args, **kwargs):
    pipe, lock = _load_pipeline(task, model)
    if pipe is None:
        raise RuntimeError("model unavailable")
    with lock:  # pipelines are not guaranteed thread-safe across sessions
        return pipe(*args, **kwargs)

EMOTION_LABELS = ["sadness", "anger", "fear", "joy", "surprise", "neutral"]

TEXT_CUES = {
    "sadness": {
        "words": ["sad", "upset", "hurt", "cry", "crying", "lonely", "disappointed",
                  "embarrassed", "ashamed", "guilty", "miserable", "heartbroken",
                  "depressed", "down", "unhappy", "scolded", "rejected", "ignored", "failed",
                  "died", "funeral", "grieving"],
        "phrases": ["got scolded", "was scolded", "got yelled at", "i feel bad",
                    "feel terrible", "made me cry", "i got in trouble", "feel awful"]
    },
    "anger": {
        "words": ["angry", "mad", "furious", "annoyed", "irritated", "rage", "unfair",
                  "hate", "frustrated", "blamed"],
        "phrases": ["so unfair", "lost my temper", "can't stand", "got blamed"]
    },
    "fear": {
        "words": ["scared", "afraid", "anxious", "anxiety", "worried", "nervous",
                  "panic", "panicking", "overwhelmed", "stress", "stressed", "terrified"],
        "phrases": ["what if", "i am worried", "i'm worried", "under pressure"]
    },
    "joy": {
        "words": ["happy", "great", "excited", "joyful", "proud", "relieved", "love",
                  "wonderful", "amazing", "glad", "delighted", "hopeful", "passed",
                  "cleared", "won", "selected", "promoted", "aced", "celebrate", "yay", "hurray"],
        "phrases": ["so happy", "can't wait", "that went well", "feeling good"]
    },
    "surprise": {
        "words": ["surprised", "shocked", "unexpected", "astonished", "unbelievable"],
        "phrases": ["didn't expect", "out of nowhere", "i can't believe"]
    }
}

def classify_text_fallback(text):
    """Explainable keyword fallback; not a trained model."""
    cleaned = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower().replace("’", "'"))
    # Phrases that contain a misleading cue word.
    cleaned = re.sub(r"\bpassed away\b", " died ", cleaned)
    scores = {label: 0.0 for label in EMOTION_LABELS}
    evidence = {label: [] for label in EMOTION_LABELS}
    words = cleaned.split()
    negators = {"not", "no", "never", "don't", "dont", "didn't", "didnt", "isn't", "wasn't", "aren't",
                "haven't", "hasn't", "can't", "cannot", "couldn't", "won't", "hardly"}
    # A cue word right after a negation ("not happy") doesn't count for that emotion.
    tokens = {w for i, w in enumerate(words) if not negators.intersection(words[max(0, i - 2):i])}
    for emotion, cues in TEXT_CUES.items():
        for phrase in cues["phrases"]:
            if re.search(rf"\b{re.escape(phrase)}\b", cleaned):
                scores[emotion] += 2.0
                evidence[emotion].append(phrase)
        for word in cues["words"]:
            if word in tokens:
                scores[emotion] += 1.0
                evidence[emotion].append(word)
    if not any(scores.values()):
        scores["neutral"] = 1.0
        return scores, ["No strong emotion keywords found; treat this result as uncertain."]
    total = sum(scores.values())
    scores = {k: v / total for k, v in scores.items()}
    notes = [f"Text cues for {label}: {', '.join(evidence[label][:5])}."
             for label in EMOTION_LABELS if evidence[label]]
    return scores, notes

WHISPER_SR = 16000


def _whisper_model_name():
    """Whisper checkpoint. base.en is a good accuracy/speed balance on CPU; override with
    the WHISPER_MODEL secret or environment variable (e.g. openai/whisper-small.en)."""
    import os
    try:
        name = st.secrets.get("WHISPER_MODEL", "")
    except Exception:
        name = ""
    return name or os.environ.get("WHISPER_MODEL", "openai/whisper-base.en")


def prepare_speech(audio_bytes):
    """Decode a WAV clip into clean 16 kHz mono float32 for Whisper.

    Steps: mono downmix, scale to [-1, 1], remove DC offset / low hum (80 Hz
    high-pass), resample to 16 kHz, trim leading/trailing silence, and
    normalise loudness so quiet microphones are still heard.
    Returns (samples, sample_rate, seconds_of_speech).
    """
    from scipy.io import wavfile
    from scipy.signal import butter, resample_poly, sosfiltfilt

    sr, data = wavfile.read(io.BytesIO(audio_bytes))
    if data.size == 0:
        return np.zeros(0, dtype=np.float32), WHISPER_SR, 0.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    data = data - float(np.mean(data))
    if len(data) > sr // 10:
        sos = butter(4, 80, btype="highpass", fs=sr, output="sos")
        data = sosfiltfilt(sos, data).astype(np.float32)
    if sr != WHISPER_SR:
        from math import gcd
        g = gcd(int(sr), WHISPER_SR)
        data = resample_poly(data, WHISPER_SR // g, int(sr) // g).astype(np.float32)
    # Trim silence using 30 ms frame energy relative to the loudest part.
    frame = int(0.03 * WHISPER_SR)
    if len(data) > frame * 3:
        n = len(data) // frame
        energy = np.sqrt(np.mean(data[: n * frame].reshape(n, frame) ** 2, axis=1))
        voiced = np.where(energy > max(0.015 * energy.max(), 1e-4))[0]
        if voiced.size:
            start = max(0, (voiced[0] - 10) * frame)       # keep ~0.3 s padding
            end = min(len(data), (voiced[-1] + 11) * frame)
            data = data[start:end]
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1e-4:
        data = 0.9 * data / peak
    return data.astype(np.float32), WHISPER_SR, len(data) / WHISPER_SR


def transcribe_audio(audio_bytes):
    """Transcribe a recorded WAV clip with Whisper after cleaning it up."""
    try:
        from transformers import pipeline

        speech, sr, seconds = prepare_speech(audio_bytes)
        if seconds < 0.3 or float(np.max(np.abs(speech), initial=0.0)) < 1e-3:
            return "", ["The recording was silent or too short to transcribe."]

        model_name = _whisper_model_name()
        result = _run("automatic-speech-recognition", model_name,
                      {"raw": speech, "sampling_rate": sr},
                      chunk_length_s=30, generate_kwargs={"num_beams": 3})
        transcript = " ".join(str(result.get("text", "")).split())
        # Whisper sometimes "hears" stock phrases in near-silence.
        if transcript.lower().strip(" .!") in {"", "you", "thank you", "thanks for watching", "bye"}:
            return "", ["No clear speech was detected in the recording."]
        return transcript, [f"Speech transcribed with {model_name.split('/')[-1]} ({seconds:.1f}s of audio after cleanup)."]
    except Exception as exc:
        log.warning("Speech transcription failed: %s", type(exc).__name__)
        return "", ["Speech recognition is unavailable right now, so spoken words were not analysed."]


def infer_text_emotion(text):
    """Use a pretrained text model when available; fall back to transparent keyword cues."""
    if not text or not text.strip():
        return None, ["No text description provided."]
    try:
        output = _run("text-classification", TEXT_MODEL, text[:2000])[0]
        scores = {label: 0.0 for label in EMOTION_LABELS}
        for item in output:
            label = str(item["label"]).lower()
            mapped = {"joy":"joy", "sadness":"sadness", "anger":"anger",
                      "fear":"fear", "surprise":"surprise", "neutral":"neutral",
                      "disgust":"anger"}.get(label)
            if mapped:
                scores[mapped] = max(scores[mapped], float(item["score"]))
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}, [
            "Text analyzed with a pretrained English emotion classifier."
        ]
    except Exception as exc:
        log.info("Text emotion model unavailable (%s); using keyword fallback", type(exc).__name__)
        scores, notes = classify_text_fallback(text)
        notes.append("Pretrained text model unavailable; keyword fallback used.")
        return scores, notes

def infer_facial_emotion(image_bytes):
    """Run a pretrained facial-expression model, or clearly mark it unavailable."""
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        output = _run("image-classification", FACE_MODEL, image)
        scores = {label: 0.0 for label in EMOTION_LABELS}
        for item in output:
            label = str(item["label"]).lower()
            mapped = {"happy":"joy", "happiness":"joy", "sad":"sadness",
                      "sadness":"sadness", "angry":"anger", "anger":"anger",
                      "fear":"fear", "fearful":"fear", "surprise":"surprise",
                      "surprised":"surprise", "neutral":"neutral",
                      "disgust":"anger"}.get(label)
            if mapped:
                scores[mapped] = max(scores[mapped], float(item["score"]))
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}, [
            "Facial-expression model estimates visible expression, not internal emotion with certainty."
        ]
    except Exception as exc:
        log.info("Facial-expression model unavailable: %s", type(exc).__name__)
        return None, ["Facial-expression model unavailable; the estimate relies on words and voice."]

def apply_event_hint(text_scores, text, weight=0.6):
    """Blend in the emotion carried by a recognised life event ("I passed my test" -> joy).

    Pretrained classifiers sometimes read event statements as neutral; the event
    hint nudges the text scores towards the event's usual emotion.
    """
    from .replies import detect_event
    event, _ = detect_event(text)
    if not event:
        return text_scores, None
    scores = dict(text_scores or {label: 0.0 for label in EMOTION_LABELS})
    for label in EMOTION_LABELS:
        scores[label] = (1 - weight) * float(scores.get(label, 0.0))
    scores[event["emotion"]] += weight
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}, event["key"]


def combine_emotion_scores(text_scores=None, face_scores=None, audio_arousal=None):
    combined = {label: 0.0 for label in EMOTION_LABELS}
    if text_scores:
        for label, score in text_scores.items():
            combined[label] += 0.65 * float(score)
    if face_scores:
        for label, score in face_scores.items():
            combined[label] += 0.35 * float(score)
    if audio_arousal is not None and (text_scores or face_scores):
        # Audio energy is only a weak arousal cue; it cannot identify emotion by itself.
        combined["fear"] += 0.025 * float(audio_arousal)
        combined["anger"] += 0.025 * float(audio_arousal)
    total = sum(combined.values())
    if total <= 0:
        return {"sadness": 0, "anger": 0, "fear": 0, "joy": 0, "surprise": 0, "neutral": 1}
    return {k: v / total for k, v in combined.items()}

def estimate_uncertainty(text_scores=None, face_scores=None, combined=None, face_detected=True):
    """How unsure the system is about the fused emotion estimate (0 = confident, 1 = no signal).

    Combines missing modalities, disagreement between text and face, and a flat
    (low-margin) fused distribution.
    """
    if not text_scores and not face_scores:
        return 1.0
    missing = 0.0
    if not text_scores:
        missing += 0.3
    if not face_scores or not face_detected:
        missing += 0.2
    disagreement = 0.0
    if text_scores and face_scores:
        t_top = max(text_scores, key=text_scores.get)
        f_top = max(face_scores, key=face_scores.get)
        disagreement = 0.0 if t_top == f_top else 0.2
    flatness = 0.0
    if combined:
        ordered = sorted(combined.values(), reverse=True)
        margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        flatness = 0.5 * (1.0 - clip(margin / 0.5))
    return clip(missing + disagreement + flatness)


def visual_features(image_bytes):
    """Extract a coarse visual cue using OpenCV Haar cascades.
    This is NOT reliable emotion recognition; it only estimates a smile-like visual cue."""
    try:
        import cv2
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0, 1.0, ["Could not decode image; visual input marked uncertain."]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        if face_cascade.empty() or smile_cascade.empty():
            # Detector files missing (e.g. OpenCV 5): don't treat that as "no face".
            return 0.0, 0.5, ["Face-detector files unavailable; relying on the facial-expression model."]
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces) == 0:
            return 0.0, 1.0, ["No face detected; visual input marked uncertain."]
        # Analyze the largest detected face only.
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        roi = gray[y + int(h * 0.45): y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=20, minSize=(20, 10))
        smile_cue = 1.0 if len(smiles) > 0 else 0.0
        return smile_cue, 0.25, [
            f"Detected {len(faces)} face(s).",
            "Smile-like visual cue detected." if smile_cue else "No clear smile-like cue detected."
        ]
    except Exception as e:
        log.info("Visual feature extraction failed: %s", type(e).__name__)
        return 0.0, 1.0, ["Visual feature extraction unavailable."]

def audio_features(audio_bytes):
    """Estimate audio arousal from simple signal features. This is not emotion recognition."""
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.size == 0:
            return 0.0, ["Audio clip was empty."]
        peak = max(float(np.max(np.abs(data))), 1.0)
        # Normalize integer PCM to roughly [-1, 1], and handle float recordings.
        if peak > 1.5:
            data = data / peak
        else:
            data = data / max(1.0, peak)
        rms = float(np.sqrt(np.mean(data ** 2)))
        # Zero-crossing rate is a rough texture feature, not a direct emotion measure.
        zcr = float(np.mean(np.abs(np.diff(np.signbit(data)))))
        # Normalize using broad heuristic ranges; microphone gain strongly affects this.
        energy_score = clip(rms / 0.20)
        texture_score = clip(zcr / 0.25)
        arousal = clip(0.7 * energy_score + 0.3 * texture_score)
        return arousal, [
            f"Sample rate: {sr} Hz.",
            f"Relative energy feature: {energy_score:.2f}.",
            f"Zero-crossing feature: {texture_score:.2f}.",
            "Audio arousal is a rough acoustic proxy; it does not identify a person's emotion."
        ]
    except Exception as e:
        log.info("Audio decoding failed: %s", type(e).__name__)
        return 0.0, ["Could not decode the recorded audio."]

def average_emotion_scores(score_sets):
    """Average emotion scores from sampled video frames."""
    valid = [scores for scores in score_sets if scores]
    if not valid:
        return None
    return {label: float(np.mean([scores.get(label, 0.0) for scores in valid]))
            for label in EMOTION_LABELS}
