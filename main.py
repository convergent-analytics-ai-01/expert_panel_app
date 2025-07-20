# --- Imports ---
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioStreamFormat
from streamlit_webrtc import webrtc_streamer, WebRtcMode, WebRtcStreamerContext
import threading
import numpy as np
import time
from azure.cognitiveservices.speech.audio import AudioConfig # Note: PushAudioInputStream is no longer needed
import requests
from io import BytesIO
from datetime import datetime
import sys
import queue

# --- Constants for Queue Messages ---
LOG_MESSAGE = "log"
TRANSCRIPT_MESSAGE = "transcript"
# PARTIAL_TRANSCRIPT_MESSAGE is removed as it's not relevant for batch processing

st.sidebar.text(f"Running Python {sys.version}")

# --- Initialize Session State (including the queue) ---
# current_recognition_text is removed from initialization as it's not used in batch mode
for key in ["user_question", "expert_output", "history", "transcribing", "transcript_buffer", "debug_logs"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "buffer" in key or "question" in key or "output" in key else \
                                [] if key == "history" or key == "debug_logs" else False

if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue()

# --- Debug Log System (modified to use the queue) ---
def log_debug(message):
    if st.session_state.get("DEBUG", False):
        try:
            st.session_state.message_queue.put((LOG_MESSAGE, message))
            print(f"DEBUG (queued): {message}")
        except Exception as e:
            print(f"ERROR: Could not put debug message into queue: {message} - {e}")

if "DEBUG" not in st.session_state:
    st.session_state.DEBUG = False
st.session_state.DEBUG = st.sidebar.checkbox("Show Debug Logs", value=st.session_state.DEBUG)

if st.session_state.DEBUG:
    st.sidebar.markdown("#### Debug Logs")
    # current_recognition_text cleared is also removed here
    st.sidebar.button("🧹 Clear Logs", on_click=lambda: st.session_state.update(debug_logs=[]))
    st.sidebar.text_area(
        "Logs",
        value="\n".join(st.session_state.debug_logs),
        height=200,
        placeholder="Logs will appear here after events trigger a Streamlit rerun (e.g., stopping the microphone or user interaction)."
    )
    # "Live Recognition" section is removed from the UI


# --- Configuration ---
ENDPOINT_URL = "https://expertpanel-endpoint.eastus.inference.ml.azure.com/score"
API_KEY = st.secrets["expertpanel_promptflow_apikey"]
AZURE_SPEECH_KEY = st.secrets["AZURE_SPEECH_KEY"]
AZURE_SPEECH_REGION = st.secrets["AZURE_SPEECH_REGION"]

# --- Setup SpeechConfig only (REVISED for Batch) ---
def setup_speech_config():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"
    # Silence timeouts are less critical for batch but can be kept for consistency
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "3000"
    )
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "5000"
    )
    return speech_config

# --- Transcription Worker (REVISED for Batch Recognition) ---
def transcribe_webrtc(webrtc_ctx: WebRtcStreamerContext, message_queue: queue.Queue):
    """
    This function now records all audio into a buffer and then sends it for single-shot batch transcription.
    """
    message_queue.put((LOG_MESSAGE, "🎙️ Starting audio recording for batch transcription..."))
    collected_audio_bytes_list = [] # List to collect raw bytes of audio frames

    try:
        # Collect audio frames while microphone is playing
        while webrtc_ctx.state.playing:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            if not audio_frames:
                message_queue.put((LOG_MESSAGE, "⚠️ No audio frames received during recording..."))
                time.sleep(0.1) # Prevent busy-waiting
                continue

            for frame in audio_frames:
                audio_data = frame.to_ndarray()

                # Resampling logic (same as before)
                if frame.sample_rate != 16000:
                    message_queue.put((LOG_MESSAGE, f"🔄 Resampling from {frame.sample_rate} Hz to 16000 Hz for collection"))
                    if audio_data.ndim > 1:
                        audio_data = np.mean(audio_data, axis=0) # Convert to mono if stereo
                    audio_data = audio_data.astype(np.float32) # Convert to float for interpolation
                    original_indices = np.linspace(0, 1, num=len(audio_data))
                    new_length = int(len(audio_data) * 16000 / frame.sample_rate)
                    new_indices = np.linspace(0, 1, num=new_length)
                    audio_data = np.interp(new_indices, original_indices, audio_data)
                    audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16) # Convert back to int16 PCM

                collected_audio_bytes_list.append(audio_data.tobytes())
            time.sleep(0.05) # Small sleep during collection loop

        message_queue.put((LOG_MESSAGE, "🛑 Recording stopped. Preparing audio for batch transcription..."))

        if not collected_audio_bytes_list:
            message_queue.put((LOG_MESSAGE, "⚠️ No audio recorded. Cannot transcribe empty audio."))
            message_queue.put((TRANSCRIPT_MESSAGE, "")) # Send empty transcript
            return

        # Combine all collected audio bytes into a single BytesIO buffer
        full_audio_bytes = b"".join(collected_audio_bytes_list)
        audio_stream_buffer = BytesIO(full_audio_bytes)

        # Setup SpeechConfig and AudioConfig for batch recognition
        speech_config = setup_speech_config()
        audio_config = AudioConfig(stream=audio_stream_buffer)

        # Create a new recognizer for the batch
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        message_queue.put((LOG_MESSAGE, "🎙️ Sending collected audio to Azure for batch recognition..."))
        # Perform single-shot recognition
        result = recognizer.recognize_once_async().get()

        full_text = ""
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            full_text = result.text
            message_queue.put((LOG_MESSAGE, f"✅ Azure recognized (batch): '{full_text}'"))
        elif result.reason == speechsdk.ResultReason.NoMatch:
            message_queue.put((LOG_MESSAGE, "⚠️ No speech could be recognized (NoMatch)."))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            message_queue.put((LOG_MESSAGE, f"🛑 Azure batch recognition cancelled. Reason: {cancellation_details.reason}"))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                message_queue.put((LOG_MESSAGE, f"🛑 Cancellation Error Details: {cancellation_details.error_details} (Error Code: {cancellation_details.error_code})"))
            else:
                message_queue.put((LOG_MESSAGE, f"🛑 Cancellation Details: {cancellation_details.reason} (no specific error code)"))

        # Send the final transcript to the queue for the main thread to pick up
        message_queue.put((TRANSCRIPT_MESSAGE, full_text))
        message_queue.put((LOG_MESSAGE, f"✅ Final transcript sent to queue (batch): '{full_text}'"))

    except Exception as e:
        message_queue.put((LOG_MESSAGE, f"❌ Error in batch transcription thread: {e}"))
        message_queue.put((TRANSCRIPT_MESSAGE, "")) # Ensure empty transcript is sent on error
    finally:
        # No continuous recognizer or push_stream to close explicitly in batch mode
        pass


# --- UI Layout ---
st.set_page_config(page_title="Expert Agent Panel", layout="wide")
st.markdown("<h2 style='font-size:1.6rem; font-weight:600; color:#143d7a;'>Product Development Expert Panel Discussion</h2>", unsafe_allow_html=True)

# --- Process Messages from Queue (This block runs in the main Streamlit thread) ---
while not st.session_state.message_queue.empty():
    try:
        message_type, content = st.session_state.message_queue.get_nowait()
        if message_type == LOG_MESSAGE:
            st.session_state.debug_logs.append(content)
            st.session_state.debug_logs = st.session_state.debug_logs[-200:]
        elif message_type == TRANSCRIPT_MESSAGE:
            st.session_state.transcript_buffer = content
            if content:
                st.toast("📝 Voice transcription added to input box")
            else:
                st.toast("⚠️ No recognizable speech detected. Try again.", icon="⚠️")
        # PARTIAL_TRANSCRIPT_MESSAGE handling is removed here
    except queue.Empty:
        pass
    except Exception as e:
        st.error(f"Error processing message from queue: {e}")
        st.session_state.debug_logs.append(f"ERROR processing queue: {e}")


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
            # Start transcription thread
            threading.Thread(target=transcribe_webrtc, args=(webrtc_ctx, st.session_state.message_queue,), daemon=True).start()
            st.session_state.transcribing = True
    else:
        st.warning("🎤 Click START and allow microphone access.")
        if st.session_state.transcribing:
            st.session_state.transcribing = False
            # current_recognition_text is no longer used, so no need to clear it


# --- Main Input (Text Area) ---
def clear_user_question():
    st.session_state.user_question = ""
    st.session_state.transcript_buffer = ""
    # current_recognition_text is no longer used, so no need to clear it


col1, col2 = st.columns([6, 2])
with col1:
    if st.session_state.transcript_buffer:
        log_debug(f"Main UI: Appending transcript from buffer to user_question: '{st.session_state.transcript_buffer}'")
        st.session_state.user_question += " " + st.session_state.transcript_buffer
        st.session_state.transcript_buffer = ""

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

# --- Submit Button (unchanged) ---
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

# --- Display Expert Response (unchanged) ---
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
