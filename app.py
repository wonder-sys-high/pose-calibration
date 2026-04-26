import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import queue

# MediaPipeの読み込み
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# フレームを一時保存する箱（撮影用）
@st.cache_resource
def get_frame_queue():
    return queue.Queue(maxsize=1)

frame_queue = get_frame_queue()

# AIモデルの初期化（リアルタイム性を優先してモデルレベル1を使用）
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

st.title("姿勢キャリブレーション")
st.markdown("### リアルタイム・ガイド撮影")
st.info("💡 **【使い方】**\n画面の**ブルーの縦線**に、あなたの耳・肩・腰が重なるように位置を調整してください。オレンジの線がまっすぐになれば理想的です。")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # 背景の骨格点は邪魔にならないよう極細に
        mp_drawing.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=1)
        )

        # 必要な関節ポイントの取得
        landmarks = results.pose_landmarks.landmark
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        h, w, _ = img.shape
        def to_px(landmark):
            return (int(landmark.x * w), int(landmark.y * h))

        p_ear = to_px(ear)
        p_shoulder = to_px(shoulder)
        p_hip = to_px(hip)
        p_knee = to_px(knee)
        p_ankle = to_px(ankle)

        # --- ガイドラインの描画（太め設定） ---
        
        # 1. 理想の垂直ライン（ブルー：B,G,Rで指定）
        # 腰のX座標を基準に、上下に太い垂直線を引く
        cv2.line(img, (p_hip[0], 0), (p_hip[0], h), (255, 150, 50), 5)
        
        # 2. あなたの現在の骨格ライン（オレンジ：B,G,Rで指定）
        line_color = (50, 100, 255)
        line_thickness = 6
        cv2.line(img, p_ear, p_shoulder, line_color, line_thickness)
        cv2.line(img, p_shoulder, p_hip, line_color, line_thickness)
        cv2.line(img, p_hip, p_knee, line_color, line_thickness)
        cv2.line(img, p_knee, p_ankle, line_color, line_thickness)

    # 撮影用に最新の加工済みフレームをキューに保存
    if not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put(img)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# リアルタイムカメラ起動
webrtc_ctx = webrtc_streamer(
    key="pose-calibration-live", 
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

# 撮影ボタン
if webrtc_ctx.state.playing:
    st.markdown("---")
    if st.button("📸 この姿勢で撮影（保存用ボタンを表示）", type="primary"):
        if not frame_queue.empty():
            snapshot = frame_queue.get()
            is_success, buffer = cv2.imencode(".jpg", snapshot)
            if is_success:
                st.image(snapshot, channels="BGR", caption="撮影された客観データ")
                st.download_button(
                    label="📥 スマホに画像を保存する",
                    data=buffer.tobytes(),
                    file_name="pose_alignment.jpg",
                    mime="image/jpeg"
                )
