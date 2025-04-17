import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import threading
import torch, time
import torch.nn as nn
from ultralytics import YOLO
import plotly.graph_objects as go
import pandas as pd
import requests, random
import multiprocessing
from streamlit_autorefresh import st_autorefresh
import base64

st.set_page_config(page_title="Drowsy Driving Detector", layout="wide")

MAX_HISTORY_LENGTH = 200  #200 readings in history
REFRESH_INTERVAL_MS = 5000  #refresh every 5s, can change
SUSTAINED_DROWSY_THRESHOLD = 0.6  #can be changed
SUSTAINED_DROWSY_SECONDS = 3  #10 secs of sustained drowsiness

#init session state variables
if 'sustained_drowsy_start_time' not in st.session_state:
    st.session_state.sustained_drowsy_start_time = None
if 'sound_played' not in st.session_state:
    st.session_state.sound_played = False
if 'loading_start_time' not in st.session_state:
    st.session_state.loading_start_time = time.time()

#global state with locks
drowsiness_history = []
drowsy_lock = threading.Lock()

#reset drowsiness history
def reset_drowsiness_history():
    global drowsiness_history
    with drowsy_lock:
        drowsiness_history.clear()
    st.session_state.sustained_drowsy_start_time = None
    st.session_state.sound_played = False

#autoplay audio
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

#UI design styling
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
body {
  background: #f2f2f2;
  font-family: 'Montserrat', sans-serif;
  margin: 0;
  padding: 0;
}
.header {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: #ffffff;
  text-align: center;
  padding: 50px 20px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.15);
}
.header h1 {
  margin: 0;
  font-size: 3em;
  font-weight: 700;
}
.header p {
  margin-top: 10px;
  font-size: 1.3em;
  letter-spacing: 0.5px;
}
.container {
  max-width: 1200px;
  margin: 40px auto;
  background: #ffffff;
  border-radius: 15px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
  overflow: hidden;
  padding: 20px;
}
.webcam-card {
  border: 2px solid #eaeaea;
  border-radius: 15px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.05);
  height: 100%;
}
.webcam-header {
  background: #f7f7f7;
  padding: 15px;
  text-align: center;
  border-bottom: 1px solid #eaeaea;
}
.webcam-header h2 {
  margin: 0;
  font-size: 1.5em;
  color: #333;
}
.footer {
  text-align: center;
  margin-top: 20px;
  padding: 20px;
  font-size: 0.9em;
  color: #999;
}
.metric {
  margin: 10px;
}
.metrics-container {
  padding-top: 20px;
}
.stPlotlyChart, .stLineChart {
    width: 100% !important;
}

#loading {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(18, 18, 18, 0.95);
    display: flex; flex-direction: column;
    justify-content: center; align-items: center; z-index: 9999;
    font-family: 'Montserrat', sans-serif;
}
.eye-container {
    position: relative; width: 80px; height: 80px; margin-bottom: 30px;
}
.eye {
    width: 80px; height: 80px; background: #ffffff;
    border-radius: 50%; position: absolute; top: 0; left: 0;
    overflow: hidden;
}
.pupil {
    width: 30px; height: 30px; background: #000000;
    border-radius: 50%; position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1;
}
.eyelid {
    width: 100%; height: 100%; background: #667eea;
    position: absolute; top: -100%; left: 0;
    animation: blink 2s infinite ease-in-out;
    z-index: 2;
}
.z {
    position: absolute; font-size: 24px; color: #764ba2;
    font-weight: bold;
    animation: z-fade 3s infinite ease-out;
    user-select: none;
}
.loading-text {
    margin-top: 20px; font-size: 1.4em; color: #ffffff;
    font-weight: 400; letter-spacing: 1px;
}
.progress-bar {
    width: 200px;
    height: 4px;
    background-color: rgba(255, 255, 255, 0.2);
    border-radius: 2px;
    margin-top: 20px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    width: 0%;
    background-color: #667eea;
    animation: load 6s linear forwards;
}

@keyframes blink {
    0%, 80% { top: -100%; }
    85% { top: 0; }
    95% { top: 0; }
    100% { top: -100%; }
}
@keyframes z-fade {
    0% { opacity: 0; transform: translateY(0) scale(0.8); }
    50% { opacity: 0.8; transform: translateY(-15px) scale(1); }
    100% { opacity: 0; transform: translateY(-30px) scale(1.2); }
}
@keyframes load {
    0% { width: 0%; }
    100% { width: 100%; }
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

#check if models are loaded
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    #loading screen
    st.markdown("""
    <div id="loading">
        <div class="eye-container">
            <div class="eye">
                <div class="pupil"></div>
                <div class="eyelid"></div>
            </div>
            <div class="z" style="top: 0px; right: -15px;">Z</div>
            <div class="z" style="top: -15px; right: -5px;">z</div>
            <div class="z" style="top: -30px; right: 5px;">z</div>
        </div>
        <div class="loading-text">Loading models...</div>
        <div class="progress-bar">
            <div class="progress-fill"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    #load models
    @st.cache_resource
    def load_models():
        """Loads the YOLO models."""
        try:
            drowsiness_history = []
            face_model = YOLO('models/yolov11n-face.pt')
            yolo_model_base = torch.load("models/drowsymodel.pth")
            yolo_model_base.eval()
            return face_model, yolo_model_base, drowsiness_history
        except Exception as e:
            print(f"Error loading models: {e}")
            return None, None, None

    face_model, yolo_model, drowsiness_history = load_models()

    
    #if models are loaded but minimum display time hasn't passed, wait
    if face_model is not None and yolo_model is not None:
        #set loaded flag
        st.session_state.models_loaded = True
        st.session_state.face_model = face_model
        st.session_state.yolo_model = yolo_model
        st.session_state.drowsiness_history = drowsiness_history
        st.rerun()  #rerun to display the main app
    else:
        st.error("Failed to load models.")
else:
    #reset loading start time for next session
    st.session_state.loading_start_time = time.time()
    
    #assign models from session state
    face_model = st.session_state.face_model
    yolo_model = st.session_state.yolo_model
    drowsiness_history = st.session_state.drowsiness_history

    #header
    st.markdown("""
    <div class="header">
      <h1>Drowsy Driving Detector</h1>
      <p>Real Time Analysis & Visual Metrics</p>
    </div>
    """, unsafe_allow_html=True)

    #helper funcs
    def preprocess_face(face, input_size=(224, 224)):
        """Resizes, converts color, normalizes, and converts face image to tensor."""
        try:
            face_resized = cv2.resize(face, input_size)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb.astype(np.float32) / 255.0
            tensor = torch.tensor(face_normalized).permute(2, 0, 1).unsqueeze(0)
            return tensor
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None

    #utility functions
    def compute_rolling_avg(values, window=20):
        """Computes rolling average with a given window size."""
        if not values:
            return []
        s = pd.Series(values)
        return s.rolling(window=min(len(values), window), min_periods=1).mean().tolist()

    #video processing
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model_initialized = face_model is not None and yolo_model is not None
            if not self.model_initialized:
                print("WARNING: Models not loaded correctly in VideoTransformer.")

        def transform(self, frame):
            global drowsiness_history
            if not self.model_initialized:
                img = frame.to_ndarray(format="bgr24")
                cv2.putText(img, "Models not loaded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return img

            img = frame.to_ndarray(format="bgr24")
            drowsy_value = None

            try:
                results = face_model(img, verbose=False, conf=0.5)
                if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    faces = results[0].boxes
                    max_drowsy_value_for_frame = -1.0
                    for face in faces:
                        x1, y1, x2, y2 = map(int, face.xyxy[0].numpy())
                        conf = face.conf.numpy()[0]
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        face_img = img[y1:y2, x1:x2]
                        if face_img is None or face_img.size == 0:
                            continue
                        face_tensor = preprocess_face(face_img)
                        if face_tensor is None:
                            continue
                        with torch.no_grad():
                            outputs = yolo_model(face_tensor)
                            value = outputs[0].cpu().numpy()[0][0]
                        if value > max_drowsy_value_for_frame:
                            max_drowsy_value_for_frame = value
                        label_text = f"Drowsy: {value:.2f}" if value > 0.6 else f"Not drowsy: {value:.2f}"
                        color = (0, 0, 255) if value > 0.6 else (0, 255, 0)
                        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if max_drowsy_value_for_frame > -1.0:
                        drowsy_value = max_drowsy_value_for_frame

                if drowsy_value is not None:
                    with drowsy_lock:
                        drowsiness_history.append(drowsy_value)
                        if len(drowsiness_history) > MAX_HISTORY_LENGTH:
                            del drowsiness_history[0]

            except Exception as e:
                print(f"Error during video transform: {e}")
            return img

    #streamlit layout
    col_metrics, col_webcam = st.columns(2)

    with col_webcam:
        if face_model is not None and yolo_model is not None:
            webrtc_ctx = webrtc_streamer(
                key="drowsiness-detection",
                video_processor_factory=VideoTransformer,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
        else:
            st.error("Models failed to load. Cannot start webcam stream.")

    with col_metrics:
        st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; margin-bottom: 15px;'>Drowsiness Metrics</h2>", unsafe_allow_html=True)

        #reset button w/ tooltip
        if st.button("Reset History", help="Clear all drowsiness history data and start fresh"):
            reset_drowsiness_history()
            st.success("History reset successfully!")

        alert_placeholder = st.empty()
        sustained_alert_placeholder = st.empty()
        sound_placeholder = st.empty()
        gauge_placeholder = st.empty()
        history_placeholder = st.empty()

        _ = st_autorefresh(interval=REFRESH_INTERVAL_MS, limit=None, key="metricrefresh")

        with drowsy_lock:
            history_copy = drowsiness_history.copy()

        latest_value = history_copy[-1] if history_copy else 0.0

        if latest_value > 0.9:
            alert_placeholder.warning("WARNING: High drowsiness level detected! Please take a break immediately!", icon="⚠️")
            #autoplay_audio("alert_sound.mp3")
        else:
            alert_placeholder.empty()

        #sustained drowsiness above threshold
        if latest_value > SUSTAINED_DROWSY_THRESHOLD:
            if st.session_state.sustained_drowsy_start_time is None:
                st.session_state.sustained_drowsy_start_time = time.time()
                st.session_state.sound_played = False

            elapsed_time = time.time() - st.session_state.sustained_drowsy_start_time
            if elapsed_time >= SUSTAINED_DROWSY_SECONDS and not st.session_state.sound_played:
                sustained_alert_placeholder.error(
                    f"DANGER: Sustained drowsiness detected for {SUSTAINED_DROWSY_SECONDS} seconds! Take a break now!",
                    icon="🚨"
                )
                autoplay_audio("alert_sound.mp3")
                st.session_state.sound_played = True
        else:
            st.session_state.sustained_drowsy_start_time = None
            sustained_alert_placeholder.empty()

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0.0, 1.0], 'tickwidth': 1, 'tickcolor': "darkblue"},
                   'bar': {'color': "rgba(0,0,0,0)"},
                   'bgcolor': "white",
                   'borderwidth': 2,
                   'bordercolor': "#cccccc",
                   'steps': [
                       {'range': [0.0, 0.6], 'color': "#90ee90"},
                       {'range': [0.6, 0.8], 'color': "#ffe48a"},
                       {'range': [0.8, 1.0], 'color': "#f08080"}
                   ],
                   'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }}
        ))
        gauge_fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        gauge_placeholder.plotly_chart(gauge_fig, use_container_width=True)

        if history_copy:
            df = pd.DataFrame({
                "Reading Index": range(len(history_copy)),
                "Drowsiness": history_copy
            })
            window_size = 20
            df["Rolling Avg (20 readings)"] = compute_rolling_avg(history_copy, window_size)
            history_placeholder.line_chart(df.set_index("Reading Index")[["Drowsiness", "Rolling Avg (20 readings)"]])
        else:
            history_placeholder.info("Waiting for drowsiness data...")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
      © Drowsy Driving Detector Application
    </div>
    """, unsafe_allow_html=True)