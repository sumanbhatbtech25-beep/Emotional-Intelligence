"""Emotion cue extraction and pretrained text, face, and speech inference."""
import io
import re
import numpy as np
import streamlit as st
from .learning import clip

EMOTION_LABELS = ["sadness", "anger", "fear", "joy", "surprise", "neutral"]

TEXT_CUES = {
    "sadness": {
        "words": ["sad", "upset", "hurt", "cry", "crying", "lonely", "disappointed",
                  "embarrassed", "ashamed", "guilty", "miserable", "heartbroken",
                  "depressed", "down", "unhappy", "scolded", "rejected", "ignored"],
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
                  "wonderful", "amazing", "glad", "delighted", "hopeful"],
        "phrases": ["so happy", "can't wait", "that went well", "feeling good"]
    },
    "surprise": {
        "words": ["surprised", "shocked", "unexpected", "astonished", "unbelievable"],
        "phrases": ["didn't expect", "out of nowhere", "i can't believe"]
    }
}

def classify_text_fallback(text):
    """Explainable keyword fallback; not a trained model."""
    cleaned = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower())
    scores = {label: 0.0 for label in EMOTION_LABELS}
    evidence = {label: [] for label in EMOTION_LABELS}
    tokens = set(cleaned.split())
    for emotion, cues in TEXT_CUES.items():
        for phrase in cues["phrases"]:
            if phrase in cleaned:
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

def transcribe_audio(audio_bytes):
    """Transcribe a recorded WAV clip using a pretrained speech-recognition model."""
    try:
        from scipy.io import wavfile
        from transformers import pipeline

        if "speech_recognition_pipeline" not in st.session_state:
            with st.spinner("Loading speech-to-text model for first use..."):
                st.session_state.speech_recognition_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny.en"
                )

        sr, data = wavfile.read(io.BytesIO(audio_bytes))
        if data.size == 0:
            return "", ["The recorded audio was empty."]
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            scale = float(max(abs(info.min), abs(info.max)))
            data = data.astype(np.float32) / scale
        else:
            data = data.astype(np.float32)
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            if peak > 1.0:
                data = data / peak

        result = st.session_state.speech_recognition_pipeline(
            {"raw": data, "sampling_rate": int(sr)}, chunk_length_s=20
        )
        transcript = str(result.get("text", "")).strip()
        if transcript:
            return transcript, ["Speech was transcribed automatically with Whisper Tiny English."]
        return "", ["Speech-to-text returned an empty transcript; emotion-from-words was skipped."]
    except Exception as exc:
        return "", [
            "Automatic speech transcription is unavailable, so spoken-word emotion analysis was skipped.",
            f"Speech-to-text error: {type(exc).__name__}. Check model download, memory, and audio format."
        ]

def infer_text_emotion(text):
    """Use a pretrained text model when available; fall back to transparent keyword cues."""
    if not text or not text.strip():
        return None, ["No text description provided."]
    try:
        from transformers import pipeline
        if "text_emotion_pipeline" not in st.session_state:
            with st.spinner("Loading the text emotion model for first use..."):
                st.session_state.text_emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None
                )
        output = st.session_state.text_emotion_pipeline(text[:2000])[0]
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
        scores, notes = classify_text_fallback(text)
        notes.append(f"Pretrained text model unavailable ({type(exc).__name__}); keyword fallback used.")
        return scores, notes

def infer_facial_emotion(image_bytes):
    """Run a pretrained facial-expression model, or clearly mark it unavailable."""
    try:
        from PIL import Image
        from transformers import pipeline
        if "face_emotion_pipeline" not in st.session_state:
            with st.spinner("Loading the facial-expression model for first use..."):
                st.session_state.face_emotion_pipeline = pipeline(
                    "image-classification", model="trpakov/vit-face-expression"
                )
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        output = st.session_state.face_emotion_pipeline(image)
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
        return None, [
            "Facial-expression model unavailable; the basic smile detector cannot reliably detect sadness.",
            f"Model error: {type(exc).__name__}. Text analysis is still available."
        ]

def combine_emotion_scores(text_scores=None, face_scores=None, audio_arousal=None):
    combined = {label: 0.0 for label in EMOTION_LABELS}
    if text_scores:
        for label, score in text_scores.items():
            combined[label] += 0.65 * float(score)
    if face_scores:
        for label, score in face_scores.items():
            combined[label] += 0.35 * float(score)
    if audio_arousal is not None:
        # Audio energy is only a weak arousal cue; it cannot identify emotion by itself.
        combined["fear"] += 0.025 * float(audio_arousal)
        combined["anger"] += 0.025 * float(audio_arousal)
    total = sum(combined.values())
    if total <= 0:
        return {"sadness": 0, "anger": 0, "fear": 0, "joy": 0, "surprise": 0, "neutral": 1}
    return {k: v / total for k, v in combined.items()}

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
        return 0.0, 1.0, [f"Visual feature extraction unavailable: {e}"]

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
        return 0.0, [
            "Could not decode audio. Please use a WAV recording or use Demo mode.",
            f"Details: {e}"
        ]

def average_emotion_scores(score_sets):
    """Average emotion scores from sampled video frames."""
    valid = [scores for scores in score_sets if scores]
    if not valid:
        return None
    return {label: float(np.mean([scores.get(label, 0.0) for scores in valid]))
            for label in EMOTION_LABELS}
