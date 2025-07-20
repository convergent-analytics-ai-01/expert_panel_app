# --- Imports ---
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioStreamFormat
from streamlit_webrtc import webrtc_streamer, WebRtcMode, WebRtcStreamerContext
import threading
import numpy as np
import time
from azure.cognitiveservices.speech.audio import PushAudioInputStream, AudioConfig
import requests
from io import BytesIO
from datetime import datetime
import sys

st.sidebar.text(f"Running Python {sys.version}")

# --- Debug Log System ---
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

def log_debug(message):
    # Only log if debug is enabled
    if st.session_state.get("DEBUG", False):
        # Print to console (visible in Streamlit Cloud app logs)
        print(f"DEBUG: {message}")
        # Append to session state for sidebar display
        st.session_state.debug_logs.append(message)
        # Keep only the last 200 logs to prevent memory issues
        st.session_state.debug_logs = st.session_state.debug_logs[-200:]

# Ensure DEBUG state is initialized from checkbox value (important for first run)
if "DEBUG" not in st.session_state:
    st.session_state.DEBUG = False

# Optional: Enable debug output
# The value of the checkbox will persist in st.session_state.DEBUG across reruns
st.session_state.DEBUG = st.sidebar.checkbox("Show Debug Logs", value=st.session_state.DEBUG)

if st.session_state.DEBUG:
    st.sidebar.markdown("#### Debug Logs")
    # Clear logs button
    st.sidebar.button("🧹 Clear Logs", on_click=lambda: st.session_state.update(debug_logs=[]))
    # Text area for logs. Note: Logs update after a Streamlit rerun.
    st.sidebar.text_area(
        "Logs",
        value="\n".join(st.session_state.debug_logs),
        height=200,
        placeholder="Logs will appear here after events trigger a Streamlit rerun (e.g., stopping the microphone)."
    )
    # This print will appear in Streamlit Cloud's console logs,
    # helping confirm if the debug logs list itself is being populated.
    print(f"Current debug logs count: {len(st.session_state.debug_logs)}")


# --- Configuration ---
ENDPOINT_URL = "https://expertpanel-endpoint.eastus.inference.ml.azure.com/score"
# Ensure these keys are correctly set in your Streamlit secrets
API_KEY = st.secrets["expertpanel_promptflow_apikey"]
AZURE_SPEECH_KEY = st.secrets["AZURE_SPEECH_KEY"]
AZURE_SPEECH_REGION = st.secrets["AZURE_SPEECH_REGION"]

# --- Initialize Session State ---
for key in ["user_question", "expert_output", "history", "transcribing", "transcript_buffer"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "buffer" in key or "question" in key or "output" in key else [] if key == "history" else False

# --- Setup Azure Transcriber ---
def setup_transcriber(audio_config):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"
    return speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# --- Transcription Worker ---
def transcribe_webrtc(webrtc_ctx: WebRtcStreamerContext):
    format = AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    push_stream = PushAudioInputStream(stream_format=format)
    audio_config = AudioConfig(stream=push_stream)
    transcriber = setup_transcriber(audio_config)

    results = []

    def recognized_handler(evt):
        log_debug(f"✅ Azure recognized: {evt.result.text}")
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            results.append(evt.result.text)

    def stop_handler(evt):
        log_debug("🛑 Session stopped or cancelled.")
        st.session_state.transcribing = False # Update state on main thread (safe)

        # Added detailed logging for cancellation events
        if evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            log_debug(f"🛑 Cancellation Reason: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                log_debug(f"🛑 Cancellation Error Details: {cancellation_details.error_details} (Error Code: {cancellation_details.error_code})")
            elif cancellation_details.reason == speechsdk.CancellationReason.EndOfStream:
                log_debug("🛑 End of audio stream detected by Azure Speech Service.")
        else:
            log_debug(f"🛑 Stop Reason: {evt.result.reason}")


    transcriber.recognized.connect(recognized_handler)
    transcriber.session_stopped.connect(stop_handler)
    transcriber.canceled.connect(stop_handler)

    try:
        transcriber.start_continuous_recognition_async()
        log_debug("🎙️ Started Azure recognition...")

        while webrtc_ctx.state.playing:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            if not audio_frames:
                log_debug("⚠️ No audio frames received...")
                time.sleep(0.1) # Prevent busy-waiting
                continue

            for frame in audio_frames:
                audio_data = frame.to_ndarray()

                if frame.sample_rate != 16000:
                    log_debug(f"🔄 Resampling from {frame.sample_rate} Hz to 16000 Hz")
                    if audio_data.ndim > 1:
                        audio_data = np.mean(audio_data, axis=0)
                    audio_data = audio_data.astype(np.float32)
                    original_indices = np.linspace(0, 1, num=len(audio_data))
                    new_length = int(len(audio_data) * 16000 / frame.sample_rate)
                    new_indices = np.linspace(0, 1, num=new_length)
                    audio_data = np.interp(new_indices, original_indices, audio_data)
                    audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)

                audio_data = audio_data.flatten()
                push_stream.write(audio_data.tobytes())

            time.sleep(0.05) # Shorter sleep for more responsive audio processing
    except Exception as e:
        log_debug(f"❌ Error in transcription thread: {e}")
    finally:
        transcriber.stop_continuous_recognition_async().get() # Ensure stop is awaited
        push_stream.close()
        log_debug("🎙️ Azure recognition stopped and push stream closed.")


    full_text = " ".join(results).strip()
    st.session_state.transcript_buffer = full_text
    log_debug(f"✅ Final transcript: '{full_text}'")

    # Display toast messages (these will appear on the next rerun)
    if full_text:
        st.toast("📝 Voice transcription added to input box")
    else:
        st.toast("⚠️ No recognizable speech detected. Try again.", icon="⚠️")

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

    if webrtc_ctx.state.playing:
        st.success("🎙️ Microphone is live and recording...")
        if not st.session_state.transcribing:
            # Start transcription thread only if not already running
            threading.Thread(target=transcribe_webrtc, args=(webrtc_ctx,), daemon=True).start()
            st.session_state.transcribing = True
    else: # webrtc_ctx is None or webrtc_ctx.state.playing is False
        st.warning("🎤 Click START and allow microphone access.")
        # Reset transcribing state when microphone is not playing
        if st.session_state.transcribing:
            st.session_state.transcribing = False


# --- Main Input ---
def clear_user_question():
    st.session_state.user_question = ""

col1, col2 = st.columns([6, 2])
with col1:
    # This block executes on every rerun.
    # If transcript_buffer was populated by the thread, it will now be applied.
    if st.session_state.transcript_buffer:
        log_debug(f"Main UI: Applying transcript from buffer to user_question: '{st.session_state.transcript_buffer}'")
        st.session_state.user_question += " " + st.session_state.transcript_buffer
        st.session_state.transcript_buffer = "" # Clear buffer after use

    st.text_area(
        label="User Question",
        key="user_question",
        height=130,
        placeholder="Type or speak your question here...",
        help="You can also use voice input from the sidebar",
        label_visibility="collapsed"
    )
with col2:
    st.button("🧹 Clear", on_click=clear_user_question)

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
        lines = text.split("\n")
        formatted = []
        for line in lines:
            if line.startswith("Host:"):
                formatted.append("<span style='color:#745C00; font-weight:bold;'>Host:</span><br> " + line[5:].strip())
            elif line.startswith("Bill Brown:"):
                formatted.append("<span style='color:#00575D; font-weight:bold;'>👨‍💼 Bill Brown:</span><br> " + line[11:].strip())
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
