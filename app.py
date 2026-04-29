import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import queue
import time
import streamlit.components.v1 as components # 音を鳴らすために追加

# MediaPipeの読み込み
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# シャッター音を鳴らすためのJavaScript関数
def play_shutter_sound():
    # 公開されているシャッター音のURL（例）
    sound_url = "https://www.soundect.com/files/shutter.mp3" # もし動作しない場合は別のmp3URLに変更可
    components.html(
        f"""
        <audio autoplay>
          <source src="{sound_url}" type="audio/mp3">
        </audio>
        """,
        height=0,
    )

# フレームと診断用データを一時保存する箱
@st.cache_resource
def get_frame_queue():
    return queue.Queue(maxsize=1)

frame_queue = get_frame_queue()

# AIモデルの初期化
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  
)

st.title("姿勢キャリブレーション")
st.markdown("### リアルタイム・ガイド撮影")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    head_forward = 0
    body_forward = 0
    w = img.shape[1]

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=1)
        )

        landmarks = results.pose_landmarks.landmark
        p_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * img.shape[0]))
        p_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0]))
        p_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * img.shape[0]))
        p_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * img.shape[0]))
        p_ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * img.shape[0]))

        facing_left = p_ear[0] < p_shoulder[0]
        if facing_left:
            head_forward = p_shoulder[0] - p_ear[0]
            body_forward = p_hip[0] - p_shoulder[0]
        else:
            head_forward = p_ear[0] - p_shoulder[0]
            body_forward = p_shoulder[0] - p_hip[0]

        cv2.line(img, (p_hip[0], 0), (p_hip[0], img.shape[0]), (255, 150, 50), 2)
        line_color, line_thickness = (50, 100, 255), 3
        for start, end in [(p_ear, p_shoulder), (p_shoulder, p_hip), (p_hip, p_knee), (p_knee, p_ankle)]:
            cv2.line(img, start, end, line_color, line_thickness)

    if not frame_queue.empty():
        try: frame_queue.get_nowait()
        except queue.Empty: pass
    frame_queue.put((img, head_forward, body_forward, w))
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="pose-calibration-live", 
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

if webrtc_ctx.state.playing:
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        btn_now = st.button("即座に撮影する", type="primary", use_container_width=True)
    with col2:
        btn_timer = st.button("⏱ 30秒タイマー", type="secondary", use_container_width=True)

    if btn_now or btn_timer:
        if btn_timer:
            countdown_placeholder = st.empty()
            for i in range(30, 0, -1):
                countdown_placeholder.markdown(f"<h2 style='text-align: center; color: red;'>撮影まであと {i} 秒...</h2>", unsafe_allow_html=True)
                time.sleep(1)
            countdown_placeholder.empty()

        # 撮影の瞬間に音を鳴らす
        play_shutter_sound()
        # 視覚的な通知（トースト）を出す
        st.toast("カシャッ！撮影しました📸")

        if not frame_queue.empty():
            snapshot, head_forward, body_forward, img_w = frame_queue.get()
            is_success, buffer = cv2.imencode(".jpg", snapshot)
            if is_success:
                st.image(snapshot, channels="BGR", caption="撮影されたデータ")
                st.download_button(label="📥 画像を保存", data=buffer.tobytes(), file_name="pose_check.jpg", mime="image/jpeg")
                
                st.markdown("## 📊 詳細診断レポート")
                threshold = img_w * 0.04
                # (以下、診断レポートの表示ロジックは前回と同様)
                if head_forward > threshold * 1.5: st.error("⚠️ 重度：頭がかなり前に出ています")
                elif head_forward > threshold: st.warning("🟡 軽度：頭が少し前に出ています")
                else: st.success("✨ 正常：理想的な位置です")
                
                if body_forward < -threshold: st.error("⚠️ 過緊張：反り腰の状態です")
                elif body_forward > threshold: st.warning("🟡 脱力過多：猫背の状態です")
                else: st.success("✨ 正常：ニュートラルな状態です")
