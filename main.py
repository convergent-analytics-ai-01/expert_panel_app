# --- Imports ---
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioStreamFormat, PushAudioInputStream, AudioConfig
from streamlit_webrtc import webrtc_streamer, WebRtcMode, WebRtcStreamerContext
import threading
import numpy as np
import time
import requests
from io import BytesIO
from datetime import datetime
import sys
import queue

# --- Constants for Queue Messages ---
LOG_MESSAGE = "log"
TRANSCRIPT_MESSAGE = "transcript"
PARTIAL_TRANSCRIPT_MESSAGE = "partial_transcript"

st.sidebar.text(f"Running Python {sys.version}")

# --- Initialize Session State (including the queue) ---
for key in ["user_question", "expert_output", "history", "transcribing", "transcript_buffer", "debug_logs", "current_recognition_text"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "buffer" in key or "question" in key or "output" in key or key == "current_recognition_text" else \
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
    st.sidebar.button("🧹 Clear Logs", on_click=lambda: st.session_state.update(debug_logs=[], current_recognition_text=""))
    st.sidebar.text_area(
        "Logs",
        value="\n".join(st.session_state.debug_logs),
        height=200,
        placeholder="Logs will appear here after events trigger a Streamlit rerun (e.g., stopping the microphone or user interaction)."
    )
    st.sidebar.markdown("#### Live Recognition")
    st.sidebar.code(st.session_state.current_recognition_text, language="text")


# --- Configuration ---
ENDPOINT_URL = "https://expertpanel-endpoint.eastus.inference.ml.azure.com/score"
API_KEY = st.secrets["expertpanel_promptflow_apikey"]
AZURE_SPEECH_KEY = st.secrets["AZURE_SPEECH_KEY"]
AZURE_SPEECH_REGION = st.secrets["AZURE_SPEECH_REGION"]

# --- Setup Azure Transcriber (for continuous recognition - REVISED WITH NEW TIMEOUTS) ---
def setup_transcriber(audio_config, message_queue):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"

    # Set various timeout properties to allow for longer pauses and speech detection
    # This is in milliseconds. 15 seconds is quite long, for testing.
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "15000" # 15 seconds of silence before finalizing an utterance
    )
    # This timeout is for how long the service will wait for *initial* speech detection.
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "15000" # 15 seconds for speech to start
    )
    # This is crucial for continuous recognition: max audio duration without speech before it stops.
    # Set to a very high value for debugging long utterances/pauses.
    speech_config.set_property(
        speechsdk.PropertyId.FromContinuousRecognitionResult_NoEndpointSpeechDetectedTimeoutMs, "300000" # 5 minutes (300 seconds) without detecting speech endpoint
    )
    # This relates to how long it waits for speech to *begin* after connection for a segment.
    speech_config.set_property(
        speechsdk.PropertyId.FromContinuousRecognitionResult_SpeechStartDetectedTimeoutMs, "10000" # 10 seconds for speech start within a segment
    )

    return speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)


# --- Transcription Worker (for continuous recognition with Azure) ---
def transcribe_webrtc(webrtc_ctx: WebRtcStreamerContext, message_queue: queue.Queue):
    format = AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    push_stream = PushAudioInputStream(stream_format=format)
    audio_config = AudioConfig(stream=push_stream)

    transcriber = None
    results = []
    current_recognizing_text_list = [""]

    try:
        transcriber = setup_transcriber(audio_config, message_queue)

        def recognized_handler(evt):
            nonlocal current_recognizing_text_list
            message_queue.put((LOG_MESSAGE, f"✅ Azure recognized (final): '{evt.result.text}'"))
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                results.append(evt.result.text)
                current_recognizing_text_list[0] = ""
                message_queue.put((PARTIAL_TRANSCRIPT_MESSAGE, current_recognizing_text_list[0]))

        def recognizing_handler(evt):
            nonlocal current_recognizing_text_list
            message_queue.put((LOG_MESSAGE, f"📝 Azure recognizing (partial): '{evt.result.text}'"))
            current_recognizing_text_list[0] = evt.result.text
            message_queue.put((PARTIAL_TRANSCRIPT_MESSAGE, current_recognizing_text_list[0]))

        def canceled_handler(evt):
            nonlocal current_recognizing_text_list
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


        transcriber.recognized.connect(recognized_handler)
        transcriber.recognizing.connect(recognizing_handler)
        transcriber.session_stopped.connect(session_stopped_handler)
        transcriber.canceled.connect(canceled_handler)

        transcriber.start_continuous_recognition_async()
        message_queue.put((LOG_MESSAGE, "🎙️ Started Azure continuous recognition..."))

        while webrtc_ctx.state.playing:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            if not audio_frames:
                message_queue.put((LOG_MESSAGE, "⚠️ No audio frames received..."))
                time.sleep(0.1)
                continue

            for frame in audio_frames:
                audio_data = frame.to_ndarray()

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

            time.sleep(0.05)

    except Exception as e:
        message_queue.put((LOG_MESSAGE, f"❌ Error in transcription thread: {type(e).__name__}: {e}"))
    finally:
        if transcriber:
            try:
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
        message_queue.put((TRANSCRIPT_MESSAGE, full_text))
        message_queue.put((LOG_MESSAGE, f"✅ Final transcript sent to queue: '{full_text}'"))
        message_queue.put((PARTIAL_TRANSCRIPT_MESSAGE, ""))


# --- UI Layout (rest of the code is unchanged) ---
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
        elif message_type == PARTIAL_TRANSCRIPT_MESSAGE:
            st.session_state.current_recognition_text = content
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

    if webrtc_ctx is None:
        st.warning("⚠️ WebRTC component is initializing. Please wait and click 'START' if it appears.")
    elif webrtc_ctx.state.playing:
        st.success("🎙️ Microphone is live and recording...")
        if not st.session_state.transcribing:
            threading.Thread(target=transcribe_webrtc, args=(webrtc_ctx, st.session_state.message_queue,), daemon=True).start()
            st.session_state.transcribing = True
    elif webrtc_ctx.state and not webrtc_ctx.state.playing:
        st.warning("🎤 Microphone stopped. Click 'START' to re-activate.")
        if st.session_state.transcribing:
            st.session_state.transcribing = False
            st.session_state.current_recognition_text = ""
    else:
        st.warning("⚠️ WebRTC component is in an unknown state. Try refreshing the page.")


# --- Main Input (Text Area) ---
def clear_user_question():
    st.session_state.user_question = ""
    st.session_state.transcript_buffer = ""
    st.session_state.current_recognition_text = ""

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
