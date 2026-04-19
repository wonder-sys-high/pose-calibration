import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

# エラーを回避するため、MediaPipeの機能を「直接」指名して読み込む
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# MediaPipeの姿勢推定モデルを初期化
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.title("姿勢AI判定テスト (Streamlit版)")
st.write("カメラへのアクセスを許可し、全身が映るようにしてください。")

# カメラ映像の各フレームを処理する関数
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTCを使用してカメラを起動
webrtc_streamer(
    key="pose-estimation", 
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
