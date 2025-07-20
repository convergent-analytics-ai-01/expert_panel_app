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
import queue # Import the queue module for thread-safe communication

# --- Constants for Queue Messages ---
LOG_MESSAGE = "log"
TRANSCRIPT_MESSAGE = "transcript"
PARTIAL_TRANSCRIPT_MESSAGE = "partial_transcript" # For live interim results

st.sidebar.text(f"Running Python {sys.version}")

# --- Initialize Session State (including the queue) ---
# Ensure all necessary keys are initialized
for key in ["user_question", "expert_output", "history", "transcribing", "transcript_buffer", "debug_logs", "current_recognition_text"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "buffer" in key or "question" in key or "output" in key or key == "current_recognition_text" else \
                                [] if key == "history" or key == "debug_logs" else False

# Initialize the thread-safe queue in session state
if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue()

# --- Debug Log System (modified to use the queue) ---
def log_debug(message):
    if st.session_state.get("DEBUG", False):
        try:
            # Put a tuple (message_type, content) into the queue
            st.session_state.message_queue.put((LOG_MESSAGE, message))
            # Also print to console for immediate visibility in Streamlit Cloud app logs
            print(f"DEBUG (queued): {message}")
        except Exception as e:
            # Fallback print if queueing fails (e.g., during app shutdown)
            print(f"ERROR: Could not put debug message into queue: {message} - {e}")

# Optional: Enable debug output checkbox (state is persistent)
if "DEBUG" not in st.session_state:
    st.session_state.DEBUG = False # Default to False if not set
st.session_state.DEBUG = st.sidebar.checkbox("Show Debug Logs", value=st.session_state.DEBUG)

if st.session_state.DEBUG:
    st.sidebar.markdown("#### Debug Logs")
    st.sidebar.button("🧹 Clear Logs", on_click=lambda: st.session_state.update(debug_logs=[], current_recognition_text=""))
    st.sidebar.text_area(
        "Logs",
        value="\n".join(st.session_state.debug_logs),
        height=200,
        placeholder="Logs will appear here after events trigger a Streamlit rerun (e.g., stopping the microphone or user interaction)."
    )
    st.sidebar.markdown("#### Live Recognition")
    # Display partial (recognizing) results live
    st.sidebar.code(st.session_state.current_recognition_text, language="text")


# --- Configuration ---
ENDPOINT_URL = "https://expertpanel-endpoint.eastus.inference.ml.azure.com/score"
# Ensure these keys are correctly set in your Streamlit secrets
API_KEY = st.secrets["expertpanel_promptflow_apikey"]
AZURE_SPEECH_KEY = st.secrets["AZURE_SPEECH_KEY"]
AZURE_SPEECH_REGION = st.secrets["AZURE_SPEECH_REGION"]

# --- Setup Azure Transcriber ---
def setup_transcriber(audio_config):
    # This function is called from the transcription thread.
    # It sets up the speech config and recognizer.
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"
    return speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# --- Transcription Worker (runs in a separate thread) ---
def transcribe_webrtc(webrtc_ctx: WebRtcStreamerContext, message_queue: queue.Queue):
    """
    This function runs in a separate thread and handles audio capture from webrtc_ctx
    and sends it to Azure Speech-to-Text. It communicates back to the main Streamlit
    thread via a thread-safe queue.
    """
    format = AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    push_stream = PushAudioInputStream(stream_format=format)
    audio_config = AudioConfig(stream=push_stream)

    transcriber = None # Initialize transcriber to None

    try:
        transcriber = setup_transcriber(audio_config)
    except Exception as e:
        message_queue.put((LOG_MESSAGE, f"❌ Failed to set up Azure Speech Recognizer: {e}"))
        message_queue.put((TRANSCRIPT_MESSAGE, "")) # Signal transcription failure
        return # Exit the thread early if setup fails

    results = []
    # Using a list to hold the recognizing text so it can be updated by nested functions (nonlocal behavior)
    current_recognizing_text_list = [""]

    def recognized_handler(evt):
        message_queue.put((LOG_MESSAGE, f"✅ Azure recognized (final): '{evt.result.text}'"))
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            results.append(evt.result.text)
            current_recognizing_text_list[0] = "" # Clear partial text after a final recognition
            message_queue.put((PARTIAL_TRANSCRIPT_MESSAGE, current_recognizing_text_list[0])) # Clear live display

    def recognizing_handler(evt):
        # This event provides interim results as speech is ongoing
        message_queue.put((LOG_MESSAGE, f"📝 Azure recognizing (partial): '{evt.result.text}'"))
        current_recognizing_text_list[0] = evt.result.text
        message_queue.put((PARTIAL_TRANSCRIPT_MESSAGE, current_recognizing_text_list[0]))

    def canceled_handler(evt):
        message_queue.put((LOG_MESSAGE, "🛑 Azure recognition session cancelled."))
        if evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            message_queue.put((LOG_MESSAGE, f"🛑 Cancellation Reason: {cancellation_details.reason}"))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                message_queue.put((LOG_MESSAGE, f"🛑 Cancellation Error Details: {cancellation_details.error_details} (Error Code: {cancellation_details.error_code})"))
            elif cancellation_details.reason == speechsdk.CancellationReason.EndOfStream:
                message_queue.put((LOG_MESSAGE, "🛑 End of audio stream detected by Azure Speech Service (possible silence timeout)."))
        message_queue.put((LOG_MESSAGE, "🛑 Signalling transcription stop."))
        current_recognizing_text_list[0] = ""
        message_queue.put((PARTIAL_TRANSCRIPT_MESSAGE, current_recognizing_text_list[0]))

    def session_stopped_handler(evt):
        message_queue.put((LOG_MESSAGE, "🛑 Azure recognition session stopped."))

    # Connect event handlers
    transcriber.recognized.connect(recognized_handler)
    transcriber.recognizing.connect(recognizing_handler) # Connect to get partial results
    transcriber.session_stopped.connect(session_stopped_handler)
    transcriber.canceled.connect(canceled_handler)

    try:
        transcriber.start_continuous_recognition_async()
        message_queue.put((LOG_MESSAGE, "🎙️ Started Azure recognition..."))

        # Loop while the WebRTC microphone is active
        while webrtc_ctx.state.playing:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            if not audio_frames:
                message_queue.put((LOG_MESSAGE, "⚠️ No audio frames received..."))
                time.sleep(0.1) # Prevent busy-waiting if no frames
                continue

            for frame in audio_frames:
                audio_data = frame.to_ndarray()

                # Resampling logic (same as before)
                if frame.sample_rate != 16000:
                    message_queue.put((LOG_MESSAGE, f"🔄 Resampling from {frame.sample_rate} Hz to 16000 Hz"))
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

            time.sleep(0.05) # Small sleep to yield CPU and process frames smoothly

    except Exception as e:
        message_queue.put((LOG_MESSAGE, f"❌ Error in transcription thread: {e}"))
    finally:
        # Ensure Azure recognition is stopped and audio stream is closed gracefully
        if transcriber: # Only try to stop if transcriber was successfully created
            try:
                # .get() blocks until the async operation completes, ensuring clean shutdown
                transcriber.stop_continuous_recognition_async().get()
                message_queue.put((LOG_MESSAGE, "🎙️ Azure recognition explicitly stopped."))
            except Exception as e:
                message_queue.put((LOG_MESSAGE, f"❌ Error stopping Azure recognition explicitly: {e}"))

        try:
            push_stream.close()
            message_queue.put((LOG_MESSAGE, "🎙️ Push audio stream closed."))
        except Exception as e:
            message_queue.put((LOG_MESSAGE, f"❌ Error closing push stream: {e}"))

        full_text = " ".join(results).strip()
        # Send the final transcript to the queue for the main thread to pick up
        message_queue.put((TRANSCRIPT_MESSAGE, full_text))
        message_queue.put((LOG_MESSAGE, f"✅ Final transcript sent to queue: '{full_text}'"))
        message_queue.put((PARTIAL_TRANSCRIPT_MESSAGE, "")) # Clear live display on thread exit


# --- UI Layout ---
st.set_page_config(page_title="Expert Agent Panel", layout="wide")
st.markdown("<h2 style='font-size:1.6rem; font-weight:600; color:#143d7a;'>Product Development Expert Panel Discussion</h2>", unsafe_allow_html=True)

# --- Process Messages from Queue (This block runs in the main Streamlit thread) ---
# This loop runs on every Streamlit rerun to pick up messages from the background thread
while not st.session_state.message_queue.empty():
    try:
        message_type, content = st.session_state.message_queue.get_nowait()
        if message_type == LOG_MESSAGE:
            st.session_state.debug_logs.append(content)
            st.session_state.debug_logs = st.session_state.debug_logs[-200:] # Keep last 200
        elif message_type == TRANSCRIPT_MESSAGE:
            st.session_state.transcript_buffer = content
            # Call st.toast here, as it's now in the main thread context
            if content:
                st.toast("📝 Voice transcription added to input box")
            else:
                st.toast("⚠️ No recognizable speech detected. Try again.", icon="⚠️")
        elif message_type == PARTIAL_TRANSCRIPT_MESSAGE:
            st.session_state.current_recognition_text = content
    except queue.Empty:
        # This theoretically shouldn't happen due to the while condition, but good practice
        pass
    except Exception as e:
        st.error(f"Error processing message from queue: {e}")
        st.session_state.debug_logs.append(f"ERROR processing queue: {e}") # Add to logs directly if queue processing itself fails


# --- Sidebar: Voice Input with WebRTC ---
with st.sidebar:
    st.header("🎤 Voice Input (WebRTC + Azure)")
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    webrtc_ctx = webrtc_streamer(
        key="azure-stream",
        mode=WebRtcMode.SENDONLY, # Only send audio from client to server
        audio_receiver_size=1024, # Buffer size for audio frames
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False}, # Request audio only
    )

    # Check if microphone is active
    if webrtc_ctx.state.playing:
        st.success("🎙️ Microphone is live and recording...")
        # Start transcription thread only if it's not already running
        if not st.session_state.transcribing:
            # Pass the message_queue to the thread for communication
            threading.Thread(target=transcribe_webrtc, args=(webrtc_ctx, st.session_state.message_queue,), daemon=True).start()
            st.session_state.transcribing = True # Mark as transcribing
    else: # webrtc_ctx is None (initial load) or webrtc_ctx.state.playing is False (stopped)
        st.warning("🎤 Click START and allow microphone access.")
        # Reset transcribing state when microphone is not playing
        if st.session_state.transcribing:
            st.session_state.transcribing = False
            # Clear any leftover partial text if mic stops
            st.session_state.current_recognition_text = ""


# --- Main Input (Text Area) ---
def clear_user_question():
    st.session_state.user_question = ""
    # Also clear transcript buffer and live recognition on clear
    st.session_state.transcript_buffer = ""
    st.session_state.current_recognition_text = ""


col1, col2 = st.columns([6, 2])
with col1:
    # This block executes on every rerun.
    # If transcript_buffer was populated by the queue processor, it will now be applied.
    if st.session_state.transcript_buffer:
        log_debug(f"Main UI: Appending transcript from buffer to user_question: '{st.session_state.transcript_buffer}'")
        # Append to existing user question to allow multi-segment input
        st.session_state.user_question += " " + st.session_state.transcript_buffer
        st.session_state.transcript_buffer = "" # Clear buffer after use

    st.text_area(
        label="User Question",
        key="user_question", # This key links the widget to session_state.user_question
        height=130,
        placeholder="Type or speak your question here...",
        help="You can also use voice input from the sidebar",
        label_visibility="collapsed" # Hide the default label for a cleaner look
    )
with col2:
    # --- FIX HERE: Changed 'on_on_click' to 'on_click' ---
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
            response.raise_for_status() # Raise an exception for HTTP errors
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
