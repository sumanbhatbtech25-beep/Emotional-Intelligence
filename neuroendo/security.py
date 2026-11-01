"""Security and privacy helpers: output escaping, login throttling, data wiping."""
import hmac
import html
import logging
import threading
import time

import streamlit as st

log = logging.getLogger("neuroendo")

MAX_TEXT_CHARS = 1000            # longest message accepted from a user
IDLE_MINUTES = 30                # personal data is wiped after this much inactivity
LOGIN_MAX_FAILURES = 5           # failed passwords allowed per client...
LOGIN_WINDOW_SECONDS = 15 * 60   # ...within this window, before a lockout of the same length
SESSION_MAX_HOURS = 8            # a sign-in lasts at most this long

# Keys in st.session_state that hold anything the user said or recorded.
PERSONAL_KEYS = ("messages", "chat_capture", "transcript", "convo_log", "emotion_notes",
                 "last_notes", "last_features", "emotion_scores", "current_log_id",
                 "last_x", "last_probs", "last_scores", "last_pred", "emotion_source")


# ------------------------------------------------------------------ escaping
def esc_html(text):
    """Escape text for use inside unsafe_allow_html markup."""
    return html.escape(str(text or ""), quote=True)


def plain_text_html(text, prefix="", suffix="", css_class="neo-said"):
    """Render user text as literal plain text.

    The text is HTML-escaped and wrapped in an HTML block, which Streamlit's
    markdown renderer leaves alone: no links, images (which would make the
    browser contact another server), auto-linked URLs or formatting.
    Use with st.markdown(..., unsafe_allow_html=True).
    """
    return f'<div class="{css_class}">{esc_html(prefix)}{esc_html(text)}{esc_html(suffix)}</div>'


def clean_text(text, limit=MAX_TEXT_CHARS):
    """Trim, collapse whitespace, drop control characters and cap the length."""
    text = "".join(ch for ch in str(text or "") if ch.isprintable() or ch in "\n\t")
    return " ".join(text.split())[:limit]


# ------------------------------------------------------------- data wiping
def wipe_personal_data(keep_learning=True):
    """Delete everything the user said or recorded from this session.

    The learned weights and hormone levels hold no personal content, so they
    are kept unless keep_learning=False.
    """
    for key in PERSONAL_KEYS:
        st.session_state.pop(key, None)
    buf = st.session_state.get("live_capture_buffer")
    if buf is not None:
        buf.wipe()
    if not keep_learning:
        for key in ("W", "reward", "stress", "history", "ab_results", "reply_variant"):
            st.session_state.pop(key, None)


def enforce_idle_timeout():
    """Wipe personal data if the session has been idle too long. Returns True if wiped."""
    now = time.time()
    last = st.session_state.get("_last_activity")
    st.session_state["_last_activity"] = now
    if last is not None and now - last > IDLE_MINUTES * 60:
        wipe_personal_data()
        return True
    return False


def cap_list(key, limit):
    """Keep session lists bounded so one session can't grow without limit."""
    items = st.session_state.get(key)
    if isinstance(items, list) and len(items) > limit:
        del items[: len(items) - limit]


# ------------------------------------------------------------ login throttle
@st.cache_resource
def _login_attempts():
    """Shared across sessions: {client_key: [failure timestamps]}."""
    return {}, threading.Lock()


def _client_key():
    """Client IP when Streamlit provides a real one, else a shared fallback key."""
    try:
        ip = st.context.ip_address
    except Exception:
        ip = None
    return ip if isinstance(ip, str) and ip else "unknown-client"


def _session_failures():
    now = time.time()
    recent = [t for t in st.session_state.get("_login_failures", []) if now - t < LOGIN_WINDOW_SECONDS]
    st.session_state["_login_failures"] = recent
    return recent


def login_locked_for():
    """Seconds until this client may try again (0 if not locked).

    Failures are counted per client IP (shared across sessions, so opening a
    new tab doesn't reset them) and per browser session as a second layer.
    """
    attempts, lock = _login_attempts()
    now = time.time()
    with lock:
        recent = [t for t in attempts.get(_client_key(), []) if now - t < LOGIN_WINDOW_SECONDS]
        attempts[_client_key()] = recent
    for failures in (recent, _session_failures()):
        if len(failures) >= LOGIN_MAX_FAILURES:
            return int(LOGIN_WINDOW_SECONDS - (now - failures[0])) + 1
    return 0


def check_password(candidate, expected):
    """Constant-time comparison that records failures for throttling."""
    ok = hmac.compare_digest(str(candidate).encode(), str(expected).encode())
    attempts, lock = _login_attempts()
    with lock:
        if ok:
            attempts.pop(_client_key(), None)
        else:
            attempts.setdefault(_client_key(), []).append(time.time())
    if ok:
        st.session_state.pop("_login_failures", None)
    else:
        st.session_state.setdefault("_login_failures", []).append(time.time())
        log.warning("Failed sign-in attempt")
        time.sleep(1.0)  # slows down guessing
    return ok
