# NEUROENDO architecture

## Responsibilities

- **`app.py`** — Streamlit entry point, theme, navigation, input controls, dashboard, and user feedback.
- **`neuroendo/learning.py`** — session initialization, Hebbian weight updates, simulated endocrine modulation, response generation, and the synthetic A/B experiment.
- **`neuroendo/emotion.py`** — speech transcription, text and facial-expression inference, visual/audio feature extraction, and cue fusion.
- **`neuroendo/media.py`** — low-resolution webcam capture, microphone buffering, video-frame sampling, and audio extraction.
- **`requirements.txt`** — runtime dependencies.

## Data flow

1. The user chooses simulated cues or records through the browser's webcam and microphone.
2. The media layer samples frames and buffers audio.
3. The emotion layer extracts cues and transcribes speech when possible.
4. The UI presents the estimate and lets the user select a feedback target.
5. The learning layer updates the session's Hebbian weights using the endocrine modulation factor.
6. Model Insights visualizes the weights and learning history.

## State and persistence

The current learned matrix and interaction history are stored in Streamlit session state. They are reused by later rounds in the same session but are not written to a persistent database or file. The pretrained text, face, and speech models are inference models; the feedback action does not fine-tune their parameters.

## Performance choices

- Request a modest browser capture resolution and frame rate.
- Downscale frames before encoding and keep only a small number of samples.
- Cap microphone buffering at 30 seconds.
- Run heavier emotion inference after recording stops, not on every incoming frame.

## Safety notes

Emotion estimates are uncertain signals, not diagnoses. Only record people who have provided informed consent. Do not store private recordings or credentials in the repository.
