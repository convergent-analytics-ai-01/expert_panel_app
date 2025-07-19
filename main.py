# --- Imports ---
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from streamlit_webrtc import webrtc_streamer, WebRtcMode, WebRtcStreamerContext
import av
import threading
import pydub
import time
from azure.cognitiveservices.speech.audio import PushAudioInputStream, AudioConfig
import requests
import re
from io import BytesIO
from datetime import datetime
import sys

st.sidebar.text(f"Running Python {sys.version}")


# --- Configuration ---
ENDPOINT_URL = "https://expertpanel-endpoint.eastus.inference.ml.azure.com/score"
API_KEY = st.secrets["expertpanel_promptflow_apikey"]
AZURE_SPEECH_KEY = st.secrets["AZURE_SPEECH_KEY"]
AZURE_SPEECH_REGION = st.secrets["AZURE_SPEECH_REGION"]

# --- Initialize Session State ---
if "user_question" not in st.session_state:
    st.session_state.user_question = ""
if "expert_output" not in st.session_state:
    st.session_state.expert_output = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "transcribing" not in st.session_state:
    st.session_state.transcribing = False

# --- Setup Azure Transcriber ---
def setup_transcriber(audio_config):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"
    return speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# --- Transcription Worker ---
def transcribe_webrtc(webrtc_ctx: WebRtcStreamerContext):
    push_stream = PushAudioInputStream()
    audio_config = AudioConfig(stream=push_stream)
    transcriber = setup_transcriber(audio_config)

    results = []

    def recognized_handler(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            results.append(evt.result.text)

    def stop_handler(evt):
        st.session_state.transcribing = False

    transcriber.recognized.connect(recognized_handler)
    transcriber.session_stopped.connect(stop_handler)
    transcriber.canceled.connect(stop_handler)

    transcriber.start_continuous_recognition_async()

    while webrtc_ctx.state.playing:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        if not audio_frames:
            continue

        segment = pydub.AudioSegment.empty()
        for frame in audio_frames:
            audio = pydub.AudioSegment(
                data=frame.to_ndarray().tobytes(),
                sample_width=frame.format.bytes,
                frame_rate=frame.sample_rate,
                channels=len(frame.layout.channels),
            )
            segment += audio

        segment = segment.set_channels(1).set_frame_rate(16000)
        push_stream.write(segment.raw_data)
        time.sleep(0.1)

    transcriber.stop_continuous_recognition()
    push_stream.close()

    full_text = " ".join(results)
    st.session_state.user_question += " " + full_text.strip()
    st.rerun()

# --- UI Layout ---
st.set_page_config(page_title="Expert Agent Panel", layout="wide")
st.markdown("<h2 style='font-size:1.6rem; font-weight:600; color:#143d7a;'>Product Development Expert Panel Discussion</h2>", unsafe_allow_html=True)

# --- Sidebar: Voice Input with WebRTC ---
with st.sidebar:
    st.header("🎤 Voice Input (WebRTC + Azure)")
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    webrtc_ctx = webrtc_streamer(
        key="azure-stream",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_ctx and webrtc_ctx.state.playing:
        st.success("🎙️ Microphone is live and recording...")
        if not st.session_state.transcribing:
            threading.Thread(target=transcribe_webrtc, args=(webrtc_ctx,), daemon=True).start()
            st.session_state.transcribing = True
    elif webrtc_ctx:
        st.warning("🎤 Click START and allow microphone access.")

# --- Main Input ---
def clear_user_question():
    st.session_state.user_question = ""

col1, col2 = st.columns([6, 2])
with col1:
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

# --- Submit Button ---
if st.button("💻 Submit Question", disabled=not st.session_state.user_question.strip()):
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
        lines = text.split("\\n")
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
