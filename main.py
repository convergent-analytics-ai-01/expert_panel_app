# --- Imports ---
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
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

# --- Configuration ---
ENDPOINT_URL = "https://expertpanel-endpoint.eastus.inference.ml.azure.com/score"
API_KEY = st.secrets["expertpanel_promptflow_apikey"]
model_size = "small.en"  # Options: tiny.en, base.en, small.en, medium.en, large
compute_type = "int8"    # Options: int8, float16, float32

# --- Initialize Session State ---
if "user_question" not in st.session_state:
    st.session_state.user_question = ""
if "expert_output" not in st.session_state:
    st.session_state.expert_output = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "audio_text_buffer" not in st.session_state:
    st.session_state.audio_text_buffer = ""

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

def audio_callback(frame: av.AudioFrame) -> av.AudioFrame:
    audio = frame.to_ndarray()
    audio = audio.flatten().astype(np.float32) / 32768.0
    audio_queue.put(audio)
    return frame

def transcription_worker():
    buffer = []
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break
        buffer.extend(audio_chunk.tolist())
        if len(buffer) > 16000 * 5:  # Transcribe every ~5 seconds
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_path = tmp_wav.name
                import soundfile as sf
                sf.write(tmp_path, np.array(buffer), 16000)
            segments, _ = model.transcribe(tmp_path)
            transcript = " ".join([seg.text.strip() for seg in segments])
            os.remove(tmp_path)
            if transcript:
                st.session_state.audio_text_buffer += " " + transcript
                st.rerun()
            buffer.clear()

# --- WebRTC Streamer ---
with st.sidebar:
    st.header("🎤 Real-Time Voice Input")
    st.caption("Speak your question. Transcribed locally with Whisper.")

    webrtc_ctx = webrtc_streamer(
        key="transcriber",
        mode=WebRtcMode.SENDONLY,
        in_audio=True,
        video_processor_factory=None,
        audio_receiver_size=1024,
        sendback_audio=False,
        audio_frame_callback=audio_callback,
    )

    if webrtc_ctx.state.playing:
        threading.Thread(target=transcription_worker, daemon=True).start()
        if st.session_state.audio_text_buffer:
            st.info("🧠 Live transcription:")
            st.markdown(f"**{st.session_state.audio_text_buffer.strip()}**")


# --- Text Area ---
def clear_user_question():
    st.session_state.user_question = ""
    st.session_state.audio_text_buffer = ""

with st.container():
    col1, col2 = st.columns([6, 2])
    with col1:
        st.markdown(
            "<h2 style='font-size:1.0rem; font-weight:600;'>🎙️ Enter your question and hear from trusted product development voices:</h2>",
            unsafe_allow_html=True
        )
        st.text_area(
            label="",
            key="user_question",
            height=130,
            placeholder="Type or speak your question here...",
            help="You can also use voice input from the sidebar",
            label_visibility="collapsed"
        )
    with col2:
        if st.button("🧹 Clear", on_click=clear_user_question):
            pass

# --- Add Live Transcript to User Question ---
if st.session_state.audio_text_buffer and not st.session_state.user_question:
    st.session_state.user_question = st.session_state.audio_text_buffer.strip()

# --- Submit Button ---
submit_disabled = not st.session_state.user_question.strip()
if st.button("💻 Submit Question", disabled=submit_disabled):
    with st.spinner("Gathering expert insights..."):
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }
            payload = {"user_question": st.session_state.user_question}
            response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            expert_response = result.get("expert_response_output", str(result))
            st.session_state.expert_output = expert_response
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.insert(0, {
                "question": st.session_state.user_question.strip(),
                "response": expert_response,
                "time": timestamp
            })
            st.success("✅ Response received.")
        except Exception as e:
            st.error(f"❌ Failed to get expert response: {str(e)}")

# --- Display Expert Response ---
if st.session_state.expert_output:
    def format_transcript(text):
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        lines = text.split("\n")
        formatted = []
        for line in lines:
            if line.startswith("Host:"):
                formatted.append("<span style='color:#745C00; font-weight:bold;'>Host:</span><br> " + line[5:].strip())
            elif line.startswith("Bill Brown:"):
                formatted.append("<span style='color:#00575D; font-weight:bold;'>👨‍💼Bill Brown:</span><br> " + line[11:].strip())
            elif line.startswith("Donald Reinertsen:"):
                formatted.append("<span style='color:#00575D; font-weight:bold;'>👨‍💼 Donald Reinertsen:</span><br> " + line[18:].strip())
            else:
                formatted.append(line.strip())
        return "<br>".join(formatted)

    st.subheader("📢 Expert Discussion")
    st.markdown(format_transcript(st.session_state.expert_output), unsafe_allow_html=True)

    buffer = BytesIO()
    buffer.write(st.session_state.expert_output.encode("utf-8"))
    buffer.seek(0)
    st.download_button("💾 Download Transcript (.txt)", buffer, file_name="expert_transcript.txt", mime="text/plain")

# --- History Panel ---
if st.session_state.history:
    with st.expander("📜 Previous Interactions", expanded=False):
        for item in st.session_state.history[:5]:
            st.button(f"⏳ {item['question']}", key=f"hist_{item['time']}", help="Display only, click has no action")
        if st.button("❌ Clear History"):
            st.session_state.history.clear()
