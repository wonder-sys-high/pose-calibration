import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import time
import queue

# エラーを回避するため、MediaPipeの機能を直接読み込む
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# 最新のフレーム（画像）を一時保存するための「箱」を用意
@st.cache_resource
def get_frame_queue():
    return queue.Queue(maxsize=1)

frame_queue = get_frame_queue()

# MediaPipeの姿勢推定モデルを初期化
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.title("姿勢キャリブレーション")
st.write("横を向いて、「一番良いと思う姿勢」を作ってください。")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
        )

        landmarks = results.pose_landmarks.landmark
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        h, w, _ = img.shape
        ear_x, ear_y = int(ear.x * w), int(ear.y * h)
        shoulder_x, shoulder_y = int(shoulder.x * w), int(shoulder.y * h)
        hip_x, hip_y = int(hip.x * w), int(hip.y * h)

        head_shift = ear_x - shoulder_x
        body_shift = shoulder_x - hip_x

        status_text = "Good Posture"
        color = (0, 255, 0)

        threshold = w * 0.05 

        if head_shift > threshold:
            status_text = "Error: Forward Head"
            color = (0, 0, 255)
        elif abs(body_shift) > threshold:
            status_text = "Error: Over-Tension"
            color = (0, 0, 255)

        cv2.line(img, (hip_x, 0), (hip_x, h), (255, 255, 0), 2)
        cv2.putText(img, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # --- 追加部分 ---
    # 線や文字が描かれた状態の最新画像を「箱」に入れる
    if not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put(img)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# カメラの表示
webrtc_ctx = webrtc_streamer(
    key="pose-estimation", 
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- 追加部分：タイマー撮影機能 ---
# カメラが起動している時だけ「撮影ボタン」を表示する
if webrtc_ctx.state.playing:
    st.markdown("---")
    if st.button("📸 30秒後に撮影する"):
        countdown_placeholder = st.empty()
        
        # 30秒のカウントダウン
        for i in range(30, 0, -1):
            countdown_placeholder.markdown(f"### 撮影まであと **{i}** 秒...")
            time.sleep(1)
            
        countdown_placeholder.markdown("### カシャ！📸")
        
        # 最新の画像を箱から取り出して画面に表示
        if not frame_queue.empty():
            snapshot = frame_queue.get()
            st.image(snapshot, channels="BGR", caption="あなたの姿勢（客観データ）")
        else:
            st.warning("画像が取得できませんでした。カメラに少し映った状態で再度お試しください。")
