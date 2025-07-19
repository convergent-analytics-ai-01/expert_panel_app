
# --- Imports ---
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
from faster_whisper import WhisperModel
import av
import queue
import threading
import tempfile
import os
import requests
import re
from datetime import datetime
from io import BytesIO
import soundfile as sf

# --- Configuration ---
ENDPOINT_URL = "https://expertpanel-endpoint.eastus.inference.ml.azure.com/score"
API_KEY = st.secrets["expertpanel_promptflow_apikey"]

model_size = "small.en"
compute_type = "int8"

# --- Initialize Session State ---
if "user_question" not in st.session_state:
    st.session_state.user_question = ""
if "expert_output" not in st.session_state:
    st.session_state.expert_output = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "audio_text_buffer" not in st.session_state:
    st.session_state.audio_text_buffer = ""
if "transcription_started" not in st.session_state:
    st.session_state.transcription_started = False

# --- Load Whisper Model ---
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return WhisperModel(model_size, compute_type=compute_type)

model = load_whisper_model()

# --- Page Setup ---
st.set_page_config(page_title="Expert Agent Panel", layout="wide")
st.markdown("""
    <style>
    .main > div:first-child { padding-top: 0rem; }
    h2, h3 { margin-bottom: 0.4rem; margin-top: 0.4rem; }
    .block-container > div:nth-child(2) { margin-top: 0rem; }
    .stMarkdown { margin-bottom: 0.2rem !important; }
    div[data-testid="column"] > div { margin-bottom: 0.2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h2 style='font-size:1.6rem; font-weight:600; color:#143d7a;'>Product Development Expert Panel Discussion</h2>",
    unsafe_allow_html=True
)

# --- Transcription Worker ---
audio_queue = queue.Queue()

def audio_callback(frame):
    audio = frame.to_ndarray()
    print(f"🔈 Audio received: {audio.shape}")
    if len(audio.shape) == 2:
        audio_mono = audio.mean(axis=1)
    else:
        audio_mono = audio
    print(f"🔈 Audio chunk received: shape={audio_mono.shape}")
    audio_queue.put(audio_mono)
    return frame

def transcription_worker():
    buffer = []
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if audio_chunk is None:
            print("📤 Manual flush triggered.")
            break
        buffer.extend(audio_chunk.tolist())
        print(f"🔄 Buffer size: {len(buffer)}")
        if len(buffer) > 16000 * 3:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_path = tmp_wav.name
                sf.write(tmp_path, np.array(buffer), 16000)

            with st.spinner("🧠 Transcribing your audio..."):
                segments, _ = model.transcribe(tmp_path)
                transcript = " ".join([seg.text.strip() for seg in segments])
                os.remove(tmp_path)

                if transcript:
                    st.session_state.audio_text_buffer += " " + transcript
                    print(f"✅ Transcription added: {transcript.strip()}")
            st.toast("✅ Transcription complete", icon="📝")
            print(f"🔁 Transcription rerun triggered")
            st.rerun()
            buffer.clear()

# --- WebRTC Streamer ---
with st.sidebar:
    st.header("🎤 Real-Time Voice Input")
    st.caption("Speak your question. Transcribed locally with Whisper.")

    rtc_config = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    webrtc_ctx = webrtc_streamer(
        key="transcriber",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_callback,
    )

    if webrtc_ctx and webrtc_ctx.state.playing:
        st.success("🎙️ Microphone is live and recording...")
        if not st.session_state.transcription_started:
            threading.Thread(target=transcription_worker, daemon=True).start()
            st.session_state.transcription_started = True
            print("🧵 Transcription thread started.")
    elif webrtc_ctx:
        st.warning("🎤 Microphone not yet active. Click START and allow mic access.")
    else:
        st.error("⚠️ WebRTC context not initialized.")

    if st.button("📝 Force Transcribe Now"):
        audio_queue.put(None)

    if st.session_state.audio_text_buffer:
        st.info("🧠 Live transcription:")
        st.markdown(f"**{st.session_state.audio_text_buffer.strip()}**")
