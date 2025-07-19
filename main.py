# --- Imports ---
import streamlit as st
import requests
import os
from io import BytesIO
from datetime import datetime
import re
import tempfile
import numpy as np
from faster_whisper import WhisperModel
import ffmpeg

# --- Configuration ---
ENDPOINT_URL = "https://expertpanel-endpoint.eastus.inference.ml.azure.com/score"
API_KEY = st.secrets["expertpanel_promptflow_apikey"]

def clear_user_question():
    st.session_state["user_question"] = ""

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

# --- Session State Initialization ---
if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""
if "expert_output" not in st.session_state:
    st.session_state.expert_output = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "audio_input_counter" not in st.session_state:
    st.session_state.audio_input_counter = 0

# --- Helper: Whisper Transcription ---
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return WhisperModel("base.en", compute_type="int8")

model = load_whisper_model()

def transcribe_audio_fasterwhisper(filepath):
    segments, _ = model.transcribe(filepath, beam_size=5)
    return " ".join([seg.text.strip() for seg in segments])

# --- Sidebar: Audio Input with faster-whisper ---
with st.sidebar:
    st.header("🎤 Voice Input")
    st.caption("Speak your question. Whisper will transcribe it locally.")

    uploaded_audio = st.audio_input(
        label="🎙️ Record Your Question",
        key=f"audio_input_{st.session_state.audio_input_counter}"
    )

    if uploaded_audio is not None:
        try:
            st.info("🧠 Transcribing audio locally...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_audio.read())
                tmp_filename = tmp_file.name

            transcription = transcribe_audio_fasterwhisper(tmp_filename)

            if transcription.strip():
                st.success("✅ Transcription Complete")
                st.session_state.user_question = (
                    f"{st.session_state.user_question.strip()} {transcription}".strip()
                )
                st.session_state.audio_input_counter += 1
                st.rerun()
            else:
                st.warning("🤔 No speech recognized.")
        except Exception as e:
            st.error(f"❌ Error during local transcription: {str(e)}")

# --- Main Area: Question Input ---
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
            pass  # The reset happens in the callback

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
            st.stop()

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

    with st.container():
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

# --- Helper to clean transcript for audio ---
def prepare_text_for_tts(text):
    text = re.sub(r"\\*\\*([^*]+)\\*\\*", r"\\1", text)
    lines = text.split("\\n")
    spoken = []
    for line in lines:
        line = line.strip()
        if re.match(r"^(Host:?|🧑‍💻 Host:)", line):
            content = re.sub(r"^(Host:?|🧑‍💻 Host:)", "", line).strip()
            spoken.append("The host says: " + content)
        elif re.match(r"^(Bill Brown:?|👨‍💼Bill Brown:)", line):
            content = re.sub(r"^(Bill Brown:?|👨‍💼Bill Brown:)", "", line).strip()
            spoken.append("Bill Brown's response is: " + content)
        elif re.match(r"^(Donald Reinertsen:?|👨‍💼 Donald Reinertsen:)", line):
            content = re.sub(r"^(Donald Reinertsen:?|👨‍💼 Donald Reinertsen:)", "", line).strip()
            spoken.append("Donald Reinertsen responds: " + content)
        else:
            spoken.append(line)
    return " ".join(spoken)

# --- Sidebar: TTS Feature ---
with st.sidebar:
    st.markdown("---")
    st.header("🔊 Audio Readout")
    voice = st.selectbox("Choose Voice:", ["en-US-JennyNeural", "en-US-GuyNeural"])
    if st.session_state.expert_output:
        if st.button("▶️ Play Response Audio"):
            try:
                speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
                speech_config.speech_synthesis_voice_name = voice
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
                text_for_audio = prepare_text_for_tts(st.session_state.expert_output)
                result = synthesizer.speak_text_async(text_for_audio).get()
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    audio_buffer = BytesIO(result.audio_data)
                    st.audio(audio_buffer, format="audio/wav")
                    st.download_button("💾 Download Audio (.wav)", data=audio_buffer, file_name="expert_audio.wav", mime="audio/wav")
                else:
                    st.error(f"TTS failed: {result.reason}")
            except Exception as e:
                st.error(f"Azure TTS Error: {str(e)}")
