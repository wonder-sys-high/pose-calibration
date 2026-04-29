import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import queue
import time

# MediaPipeの読み込み
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# カメラのフラッシュ効果（視覚的なシャッター）
def play_shutter_effect():
    st.markdown(
        """
        <style>
        @keyframes flash {
            0% { background-color: rgba(255, 255, 255, 1); }
            100% { background-color: rgba(255, 255, 255, 0); }
        }
        .flash-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 999999;
            pointer-events: none;
            animation: flash 0.8s ease-out;
        }
        </style>
        <div class="flash-overlay"></div>
        """,
        unsafe_allow_html=True
    )

# フレームと診断用データを一時保存する箱
@st.cache_resource
def get_frame_queue():
    return queue.Queue(maxsize=1)

frame_queue = get_frame_queue()

# AIモデルの初期化（レベル1）
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  
)

st.title("姿勢キャリブレーション")
st.markdown("### リアルタイム・ガイド撮影 ＆ 医学的リセット")
st.info("💡 **【使い方】**\n画面の**ブルーの縦線**に耳・肩・腰が重なるように調整してください。オレンジの線がまっすぐになれば理想的です。")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose.process(img_rgb)
    
    head_forward = 0
    body_forward = 0
    w = img.shape[1]

    if results.pose_landmarks:
        # 背景の骨格点は極細に
        mp_drawing.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=1)
        )

        landmarks = results.pose_landmarks.landmark
        h, _, _ = img.shape
        def to_px(landmark):
            return (int(landmark.x * w), int(landmark.y * h))

        p_ear = to_px(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value])
        p_shoulder = to_px(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        p_hip = to_px(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        p_knee = to_px(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
        p_ankle = to_px(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        # ズレの計算
        facing_left = p_ear[0] < p_shoulder[0] 
        if facing_left:
            head_forward = p_shoulder[0] - p_ear[0] 
            body_forward = p_hip[0] - p_shoulder[0] 
        else:
            head_forward = p_ear[0] - p_shoulder[0] 
            body_forward = p_shoulder[0] - p_hip[0] 

        # ガイドラインの描画
        cv2.line(img, (p_hip[0], 0), (p_hip[0], h), (255, 150, 50), 1)
        line_color, line_thick = (50, 100, 255), 2
        for start, end in [(p_ear, p_shoulder), (p_shoulder, p_hip), (p_hip, p_knee), (p_knee, p_ankle)]:
            cv2.line(img, start, end, line_color, line_thick)

    if not frame_queue.empty():
        try: frame_queue.get_nowait()
        except queue.Empty: pass
    frame_queue.put((img, head_forward, body_forward, w))
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="pose-calibration-final", 
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
            cp = st.empty()
            for i in range(30, 0, -1):
                cp.markdown(f"<h2 style='text-align: center; color: red;'>撮影まであと {i} 秒...</h2>", unsafe_allow_html=True)
                time.sleep(1)
            cp.empty()

        play_shutter_effect()
        st.toast("撮影完了📸")

        if not frame_queue.empty():
            snapshot, head_forward, body_forward, img_w = frame_queue.get()
            st.image(snapshot, channels="BGR", caption="撮影された客観データ")
            st.download_button(label="📥 画像を保存", data=cv2.imencode(".jpg", snapshot)[1].tobytes(), file_name="pose_check.jpg", mime="image/jpeg")

            # --- 医学的根拠に基づく詳細診断 ＆ リセットアクション ---
            st.markdown("## 📊 詳細診断 ＆ 1分間リセット")
            threshold = img_w * 0.04 

            # 1. 首の診断
            st.markdown("### ① 首・頭の状態")
            if head_forward > threshold:
                st.error("⚠️ ストレートネック（頭部前傾）")
                st.info("""
                **[span_0](start_span)[span_1](start_span)💡 1分リセット：チンイン（あご引き）エクササイズ**[span_0](end_span)[span_1](end_span)
                あごを軽く引き、後頭部を後ろに押し込み8秒キープ。これを繰り返すことで首の深層筋を活性化します。
                [span_2](start_span)さらに、大胸筋（胸）を広げるストレッチを組み合わせるとより効果的です[span_2](end_span)。
                """)
                # 
            else:
                st.success("✨ ニュートラルな首位置です")

            # 2. 腰・骨盤の診断
            st.markdown("### ② 腰・骨盤の状態")
            if body_forward < -threshold:
                st.error("⚠️ 反り腰・過緊張（骨盤前傾傾向）")
                st.info("""
                **[span_3](start_span)[span_4](start_span)💡 1分リセット：腸腰筋ストレッチ**[span_3](end_span)[span_4](end_span)
                片膝立ちになり、腰を反らさないよう注意して体重を前に移動。後ろ脚の付け根を30秒伸ばします。
                [span_5](start_span)[span_6](start_span)また、四つん這いで背中を丸める「キャットポーズ」も腰の緊張緩和に有効です[span_5](end_span)[span_6](end_span)。
                """)
                # 
            elif body_forward > threshold:
                st.warning("🟡 猫背・骨盤後傾（脱力過多）")
                st.info("""
                **[span_7](start_span)💡 1分リセット：タオルによる骨盤リセット**[span_7](end_span)
                丸めたバスタオルをお尻の骨（坐骨）のすぐ後ろに敷いて座ってください。物理的に骨盤を立てる感覚を体に覚え込ませます。
                [span_8](start_span)あわせて肩甲骨を中央にギュッと寄せる運動も行いましょう[span_8](end_span)。
                """)
                # 
            else:
                st.success("✨ 理想的なバランスです")
