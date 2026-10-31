# NEUROENDO architecture

## Responsibilities

- **`app.py`** — Streamlit entry point with two views switched from the top-right corner: **Chat** (live camera + conversation + feedback) and **Emotional Learning** (Emotion Studio, Model Insights, A/B Experiment, endocrine controller). Both share one network and endocrine state.
- **`neuroendo/learning.py`** — builds the feature vector from emotion scores, the Hebbian/anti-Hebbian network, the artificial endocrine system (D, S, plasticity, stress bias), reply templates, session state, and the simulated-user A/B experiment. Runs without Streamlit, so it can be tested on its own.
- **`neuroendo/replies.py`** — recognises life events in what the user said and words the reply for the chosen strategy.
- **`neuroendo/emotion.py`** — speech transcription, text and facial-expression inference, acoustic arousal, score fusion, and uncertainty estimation.
- **`neuroendo/media.py`** — low-resolution webcam capture, microphone buffering (de-interleaving browser stereo audio correctly), video-frame sampling, and audio extraction.
- **`requirements.txt`** — runtime dependencies.

## Data flow

1. On the chat page, the user talks to the live camera, or types a message.
2. The media layer samples frames and buffers audio.
3. The emotion layer transcribes speech, classifies text and facial expression, measures acoustic arousal, fuses the results into six emotion scores, and estimates uncertainty.
4. The learning layer builds `x = [emotion scores, arousal, uncertainty]` and scores each reply strategy with `x·W + stress bias`. The winning strategy selects the reply.
5. The user accepts the reply or picks a better strategy. `W` is updated with `ΔW = η·g(H)·x(y − p)ᵀ`.
6. The endocrine states update from the outcome (70%) and the user's emotional state (30%). They change plasticity `g(H)` and the stress bias for the next interaction.
7. Model Insights shows the weights, hormone and plasticity history, and the acceptance rate.

## State and persistence

The learned matrix, endocrine state and interaction history are stored in Streamlit session state. They are reused by later rounds in the same session but are not written to a persistent database or file. The pretrained text, face, and speech models are inference models; the feedback action does not fine-tune their parameters.

## Performance choices

- Request a modest browser capture resolution and frame rate.
- Downscale frames before encoding and keep only a small number of samples.
- Cap microphone buffering at 30 seconds.
- Run heavier emotion inference after recording stops, not on every incoming frame.

## Safety notes

Emotion estimates are uncertain signals, not diagnoses. Only record people who have provided informed consent. Do not store private recordings or credentials in the repository.
