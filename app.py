import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import queue
import time

# --- パスワード認証システム ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# 認証されていない場合はロック画面を表示して処理を止める
if not st.session_state["authenticated"]:
    st.title("🔒 姿勢キャリブレーション")
    st.info("この記事の購入者限定ツールです。Note記事内に記載されているパスワードを入力してください。")
    
    pwd = st.text_input("パスワード", type="password")
    if st.button("ロックを解除する", type="primary"):
        if pwd == "neko":
            st.session_state["authenticated"] = True
            st.rerun() # 画面をリロードしてメインアプリへ
        else:
            st.error("パスワードが間違っています。")
    
    # パスワードが通るまではここから下のコードを一切実行しない
    st.stop()

# ==========================================
# 以下、認証成功後に表示されるメインアプリ
# ==========================================

# MediaPipeの読み込み
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# カメラのフラッシュ効果
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

@st.cache_resource
def get_frame_queue():
    return queue.Queue(maxsize=1)

frame_queue = get_frame_queue()

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  
)

st.title("姿勢キャリブレーション")
st.markdown("### リアルタイム・ガイド撮影")
st.info("💡 **【使い方】**\n画面の**ブルーの縦線**に耳・肩・腰が重なるように調整してください。オレンジの線がまっすぐになれば理想的です。")

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
        h, _, _ = img.shape
        def to_px(landmark):
            return (int(landmark.x * w), int(landmark.y * h))

        p_ear = to_px(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value])
        p_shoulder = to_px(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        p_hip = to_px(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        p_knee = to_px(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
        p_ankle = to_px(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        facing_left = p_ear[0] < p_shoulder[0] 
        if facing_left:
            head_forward = p_shoulder[0] - p_ear[0] 
            body_forward = p_hip[0] - p_shoulder[0] 
        else:
            head_forward = p_ear[0] - p_shoulder[0] 
            body_forward = p_shoulder[0] - p_hip[0] 

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

            # --- 超・直感的な詳細診断 ＆ 1分リセット ---
            st.markdown("## 📊 診断結果 ＆ 1分リセット")
            threshold = img_w * 0.04 

            # 1. 首の診断
            st.markdown("### ① 首の位置")
            if head_forward > threshold:
                st.error("⚠️ 首が前に出ています（ストレートネック）")
                st.write("約5kgもある重い頭を、首の筋肉だけで必死に支えている状態です。")
                st.info("""
                **💡 1分リセット：壁ピタッ！二重あご体操**
                1. 壁を背にして立ち、「かかと・お尻・背中」を壁にくっつけます。
                2. 人差し指を「あごの先」に当て、あごをノド仏に向かって水平にグーッと押し込みます。（亀が首を引っ込める動き。上を向くのはNG！）
                3. わざと一番ひどい「二重あご」を作ったまま、後頭部を壁にくっつけて5秒キープ。これを3回！
                """)
            else:
                st.success("✨ まっすぐな良い首です！")

            # 2. 腰・骨盤の診断
            st.markdown("### ② 腰・背中の位置")
            if body_forward < -threshold:
                st.error("⚠️ 腰が反りすぎています（反り腰）")
                st.write("良い姿勢を作ろうとして、無意識に腰を反らして力んでいる状態です。")
                st.info("""
                **💡 1分リセット：赤ちゃんポーズ**
                1. 仰向けにゴロンと寝転がります。
                2. 両膝を両手で抱え込み、胸にギューッと引き寄せます。（腰が丸まるのを感じます）
                3. そのまま20秒深呼吸！ガチガチに緊張した腰が一気にリセットされます。
                """)
            elif body_forward > threshold:
                st.warning("🟡 背中が丸まっています（猫背）")
                st.write("骨盤が後ろに倒れてしまい、お腹が縮こまっている状態です。")
                st.info("""
                **💡 1分リセット：お尻にタオル作戦**
                1. バスタオルを固く丸めて「太い筒」を作ります。
                2. 椅子に座る時、お尻の【後ろ半分だけ】にタオルを踏むように敷きます。
                3. タオルが「くさび」になり、頑張らなくても勝手に背筋がピンと伸びます！
                """)
            else:
                st.success("✨ 理想的なバランスです！")
