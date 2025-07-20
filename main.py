# --- Imports ---
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioStreamFormat, PushAudioInputStream, AudioConfig # Re-import PushAudioInputStream
from streamlit_webrtc import webrtc_streamer, WebRtcMode, WebRtcStreamerContext
import threading
import numpy as np
import time
import requests
from io import BytesIO # Still useful, but for creating a specific audio stream, not directly passed
from datetime import datetime
import sys
import queue

# --- Constants for Queue Messages ---
LOG_MESSAGE = "log"
TRANSCRIPT_MESSAGE = "transcript"
AUDIO_BUFFER_MESSAGE = "audio_buffer" # Message type for collected raw audio bytes

st.sidebar.text(f"Running Python {sys.version}")

# --- Initialize Session State (including the queue) ---
for key in ["user_question", "expert_output", "history", "transcribing", "transcript_buffer", "debug_logs"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "buffer" in key or "question" in key or "output" in key else \
                                [] if key == "history" or key == "debug_logs" else False

if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue()

# --- Debug Log System ---
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
    st.sidebar.button("🧹 Clear Logs", on_click=lambda: st.session_state.update(debug_logs=[]))
    st.sidebar.text_area(
        "Logs",
        value="\n".join(st.session_state.debug_logs),
        height=200,
        placeholder="Logs will appear here after events trigger a Streamlit rerun (e.g., stopping the microphone or user interaction)."
    )
    # "Live Recognition" is removed for batch mode
    st.sidebar.markdown("#### Live Recognition")
    st.sidebar.code("", language="text", help="Live recognition is not active in batch mode.")


# --- Configuration ---
ENDPOINT_URL = "https://expertpanel-endpoint.eastus.inference.ml.azure.com/score"
API_KEY = st.secrets["expertpanel_promptflow_apikey"]
AZURE_SPEECH_KEY = st.secrets["AZURE_SPEECH_KEY"]
AZURE_SPEECH_REGION = st.secrets["AZURE_SPEECH_REGION"]

# --- Setup SpeechConfig only (for Batch Recognition) ---
# This prepares the basic SpeechConfig that will be used by the main thread.
def setup_speech_config():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"
    # Silence timeouts are less critical for batch but can be kept.
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "15000"
    )
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "15000"
    )
    return speech_config


# --- Transcription Worker (REVISED: only collects audio bytes) ---
def transcribe_webrtc(webrtc_ctx: WebRtcStreamerContext, message_queue: queue.Queue):
    """
    This function *only* collects audio bytes from webrtc_ctx
    and sends the complete byte buffer to the main Streamlit thread via the queue.
    The actual transcription call to Azure happens in the main thread.
    """
    message_queue.put((LOG_MESSAGE, "🎙️ Starting audio recording for batch transcription..."))
    collected_audio_bytes_list = [] # List to collect raw bytes of audio frames

    try:
        # Collect audio frames while microphone is playing
        while webrtc_ctx.state.playing:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            if not audio_frames:
                message_queue.put((LOG_MESSAGE, "⚠️ No audio frames received during recording, waiting..."))
                time.sleep(0.1) # Prevent busy-waiting
                continue

            for frame in audio_frames:
                audio_data = frame.to_ndarray()

                # Resampling and type conversion to 16kHz, mono, int16 PCM
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
                else: # Already 16kHz, just ensure mono and int16
                    if audio_data.ndim > 1:
                        audio_data = np.mean(audio_data, axis=0)
                    audio_data = audio_data.astype(np.int16) # Ensure it's int16

                collected_audio_bytes_list.append(audio_data.tobytes()) # Collect raw bytes
            time.sleep(0.05) # Small sleep during collection loop

        # Execution reaches here when webrtc_ctx.state.playing becomes False (user clicked STOP)
        message_queue.put((LOG_MESSAGE, "🛑 Recording loop finished. Sending audio buffer to main thread."))
        message_queue.put((LOG_MESSAGE, f"Total collected audio bytes chunks: {len(collected_audio_bytes_list)}"))

        if not collected_audio_bytes_list:
            message_queue.put((LOG_MESSAGE, "⚠️ No audio was actually recorded. Sending empty buffer."))
            message_queue.put((AUDIO_BUFFER_MESSAGE, b"")) # Send empty byte string to signal no audio
            return

        full_audio_bytes = b"".join(collected_audio_bytes_list)
        message_queue.put((LOG_MESSAGE, f"Prepared {len(full_audio_bytes)} bytes of audio. Placing in queue."))
        # Send the raw bytes of the collected audio to the main thread
        message_queue.put((AUDIO_BUFFER_MESSAGE, full_audio_bytes))

    except Exception as e:
        message_queue.put((LOG_MESSAGE, f"❌ UNCAUGHT EXCEPTION in audio collection thread: {type(e).__name__}: {e}"))
        message_queue.put((AUDIO_BUFFER_MESSAGE, b"")) # Ensure empty buffer on error
    finally:
        message_queue.put((LOG_MESSAGE, "Audio collection thread finished."))


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
            # --- ADDED LOGGING HERE ---
            log_debug(f"Main UI: Received TRANSCRIPT_MESSAGE. Content: '{content}'")
            if content:
                st.toast("📝 Voice transcription added to input box")
            else:
                st.toast("⚠️ No recognizable speech detected. Try again.", icon="⚠️")
        elif message_type == AUDIO_BUFFER_MESSAGE:
            # --- Main thread handles Azure API call here ---
            log_debug(f"Main UI: Received audio buffer ({len(content)} bytes) for transcription.")
            if not content: # Empty buffer received, no audio to transcribe
                log_debug("Main UI: Empty audio buffer, skipping Azure transcription.")
                st.session_state.transcript_buffer = "" # Ensure buffer is empty
                st.toast("⚠️ No recognizable speech detected. Try again.", icon="⚠️")
                continue

            try:
                # 1. Define audio format for PushAudioInputStream
                audio_stream_format = speechsdk.audio.AudioStreamFormat(
                    samples_per_second=16000,
                    bits_per_sample=16,
                    channels=1
                )
                # 2. Create PushAudioInputStream
                push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_stream_format)
                # 3. Create AudioConfig from PushAudioInputStream
                audio_config = AudioConfig(stream=push_stream)

                speech_config = setup_speech_config() # Get the config
                recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
                log_debug("Main UI: Sending collected audio to Azure for batch recognition via PushAudioInputStream...")

                # 4. Write all collected audio bytes to the push stream
                push_stream.write(content)
                # 5. Signal that no more data will be written to the stream
                push_stream.close()

                # 6. Perform single-shot recognition - this is a blocking call
                result = recognizer.recognize_once_async().get()

                full_text = ""
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    full_text = result.text
                    log_debug(f"✅ Azure recognized (batch, main thread): '{full_text}'")
                elif result.reason == speechsdk.ResultReason.NoMatch:
                    full_text = "" # Ensure empty text if no match
                    log_debug("⚠️ No speech could be recognized (NoMatch, main thread).")
                    if result.no_match_details:
                        log_debug(f"NoMatch details: {result.no_match_details.reason} - {result.no_match_details.error_details}")
                elif result.reason == speechsdk.ResultReason.Canceled:
                    full_text = "" # Ensure empty text if canceled
                    cancellation_details = result.cancellation_details
                    log_debug(f"🛑 Azure batch recognition cancelled (main thread). Reason: {cancellation_details.reason}")
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        log_debug(f"🛑 Cancellation Error Details: {cancellation_details.error_details} (Error Code: {cancellation_details.error_code})")
                    else:
                        log_debug(f"🛑 Cancellation Details: {cancellation_details.reason} (no specific error code)")

                st.session_state.transcript_buffer = full_text # Update session state
                if full_text: # Display toast based on actual result
                    st.toast("📝 Voice transcription added to input box")
                else:
                    st.toast("⚠️ No recognizable speech detected. Try again.", icon="⚠️")

            except Exception as e:
                log_debug(f"❌ ERROR transcribing audio in main thread: {type(e).__name__}: {e}")
                st.error(f"Failed to transcribe audio: {e}")
                st.session_state.transcript_buffer = "" # Clear on error
                st.toast("❌ Transcription failed. See debug logs.", icon="⚠️")

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
        key="azure-batch-stream", # Consistent key
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_ctx is None:
        st.warning("⚠️ WebRTC component is initializing. Please wait and click 'START' if it appears.")
    elif webrtc_ctx.state.playing:
        st.success("🎙️ Microphone is live and recording...")
        if not st.session_state.transcribing:
            # Start the thread to *collect* audio
            threading.Thread(target=transcribe_webrtc, args=(webrtc_ctx, st.session_state.message_queue,), daemon=True).start()
            st.session_state.transcribing = True
    elif webrtc_ctx.state and not webrtc_ctx.state.playing:
        st.warning("🎤 Microphone stopped. Click 'START' to re-activate.")
        if st.session_state.transcribing:
            st.session_state.transcribing = False
    else:
        st.warning("⚠️ WebRTC component is in an unknown state. Try refreshing the page.")


# --- Main Input (Text Area) ---
def clear_user_question():
    st.session_state.user_question = ""
    st.session_state.transcript_buffer = ""

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
