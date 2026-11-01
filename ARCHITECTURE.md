# NEUROENDO architecture

## Responsibilities

- **`app.py`** — Streamlit entry point with two views switched from the top-right corner: **Chat** (live camera + conversation + feedback) and **Emotional Learning** (Emotion Studio, Model Insights, A/B Experiment, endocrine controller). Both share one network and endocrine state.
- **`neuroendo/learning.py`** — builds the feature vector from emotion scores, the Hebbian/anti-Hebbian network, the artificial endocrine system (D, S, plasticity, stress bias), reply templates, session state, and the simulated-user A/B experiment. Runs without Streamlit, so it can be tested on its own.
- **`neuroendo/security.py`** — output escaping, wiping personal data, idle timeout, bounded history, and login throttling.
- **`neuroendo/replies.py`** — recognises life events in what the user said and words the reply for the chosen strategy.
- **`neuroendo/emotion.py`** — speech transcription, text and facial-expression inference, acoustic arousal, score fusion, and uncertainty estimation.
- **`neuroendo/media.py`** — memory-only webcam and microphone capture (de-interleaving browser stereo audio correctly), frames spread over the whole recording, and wiping the buffer after use.
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

Raw video frames and audio exist only in a per-session memory buffer, and are wiped as soon as a recording has been analysed (or if analysis fails). The learned matrix, endocrine state, conversation log (short titles) and transcripts are kept in Streamlit session state. They are never written to disk or a database, are deleted by **Delete my data**, logout, or 30 minutes of inactivity, and disappear when the session ends. The pretrained text, face and speech models are loaded once per server and shared between sessions; they hold no user data, and feedback never fine-tunes them.

## Performance choices

- Request a modest browser capture resolution and frame rate.
- Downscale frames before encoding and keep only a small number of samples.
- Cap microphone buffering at 30 seconds.
- Run heavier emotion inference after recording stops, not on every incoming frame.

## Safety notes

Emotion estimates are uncertain signals, not diagnoses. Only record people who have provided informed consent. Do not store private recordings or credentials in the repository.
