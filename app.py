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
st.markdown("### リアルタイム・ガイド撮影")
st.info("💡 **【使い方】**\n画面の**ブルーの縦線**に、あなたの耳・肩・腰が重なるように調整してください。オレンジの線がまっすぐになれば理想的です。")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose.process(img_rgb)
    
    # 診断用の変数を初期化
    head_forward = 0
    body_forward = 0
    w = img.shape[1]

    if results.pose_landmarks:
        # 背景の骨格点は極細に（グレー）
        mp_drawing.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=1)
        )

        # 関節ポイントの取得（左半身を基準）
        landmarks = results.pose_landmarks.landmark
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        h, _, _ = img.shape
        def to_px(landmark):
            return (int(landmark.x * w), int(landmark.y * h))

        p_ear = to_px(ear)
        p_shoulder = to_px(shoulder)
        p_hip = to_px(hip)
        p_knee = to_px(knee)
        p_ankle = to_px(ankle)

        # --- ズレの計算（左右どちらを向いていても対応するロジック） ---
        facing_left = p_ear[0] < p_shoulder[0] # 耳が肩より左にあれば「左向き」
        
        if facing_left:
            head_forward = p_shoulder[0] - p_ear[0] # プラスなら頭が前
            body_forward = p_hip[0] - p_shoulder[0] # プラスなら肩が前（猫背）
        else:
            head_forward = p_ear[0] - p_shoulder[0] # プラスなら頭が前
            body_forward = p_shoulder[0] - p_hip[0] # プラスなら肩が前（猫背）

        # --- ガイドラインの描画 ---
        # 1. 理想の垂直ライン（ブルー）
        cv2.line(img, (p_hip[0], 0), (p_hip[0], h), (255, 150, 50), 5)
        
        # 2. 現在の骨格ライン（オレンジ）
        line_color = (50, 100, 255)
        line_thickness = 7
        cv2.line(img, p_ear, p_shoulder, line_color, line_thickness)
        cv2.line(img, p_shoulder, p_hip, line_color, line_thickness)
        cv2.line(img, p_hip, p_knee, line_color, line_thickness)
        cv2.line(img, p_knee, p_ankle, line_color, line_thickness)

    # 撮影用に画像と「診断データ」をセットにしてキューに保存
    if not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put((img, head_forward, body_forward, w))

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# リアルタイムカメラ起動
webrtc_ctx = webrtc_streamer(
    key="pose-calibration-live", 
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

# 撮影・診断結果の表示
if webrtc_ctx.state.playing:
    st.markdown("---")
    st.write("📸 撮影方法を選んでください")
    
    # 2つのボタンを横並びに配置
    col1, col2 = st.columns(2)
    with col1:
        btn_now = st.button("即座に撮影する", type="primary", use_container_width=True)
    with col2:
        btn_timer = st.button("⏱ 30秒タイマー", type="secondary", use_container_width=True)

    if btn_now or btn_timer:
        if btn_timer:
            countdown_placeholder = st.empty()
            for i in range(30, 0, -1):
                # 離れていても見えるように大きく赤色で表示
                countdown_placeholder.markdown(f"<h2 style='text-align: center; color: red;'>撮影まであと {i} 秒...</h2>", unsafe_allow_html=True)
                time.sleep(1)
            countdown_placeholder.empty()

        if not frame_queue.empty():
            # キューから画像と診断データを取り出す
            snapshot, head_forward, body_forward, img_w = frame_queue.get()
            
            is_success, buffer = cv2.imencode(".jpg", snapshot)
            if is_success:
                st.image(snapshot, channels="BGR", caption="撮影されたデータ（主観と客観の比較）")
                
                # ダウンロードボタン
                st.download_button(
                    label="📥 画像を写真フォルダに保存",
                    data=buffer.tobytes(),
                    file_name="pose_check.jpg",
                    mime="image/jpeg"
                )

                # --- 診断レポート出力 ---
                st.markdown("## 📊 詳細診断レポート")
                threshold = img_w * 0.04 # 画面幅の4%をズレの許容範囲とする

                # 1. 首・頭の判定
                st.markdown("### 1. ストレートネック度（頭の突出）")
                if head_forward > threshold * 1.5:
                    st.error("⚠️ 重度：基準より頭がかなり前に出ています")
                    st.write("首の筋肉だけで重い頭（約5kg）を必死に支えている状態です。")
                elif head_forward > threshold:
                    st.warning("🟡 軽度：基準より頭が少し前に出ています")
                    st.write("画面に引き寄せられ、首に負担がかかり始めています。")
                elif head_forward < -threshold:
                    st.warning("🟡 引きすぎ：顎を引きすぎて首が詰まっています")
                else:
                    st.success("✨ 正常：理想的な位置です")
                    st.write("頭の重さが正しく分散され、首への負担が最小限に抑えられています。")

                # 2. 腰・背中の判定
                st.markdown("### 2. 腰・背中のバランス")
                if body_forward < -threshold:
                    st.error("⚠️ 過緊張：肩が腰より極端に後ろにあります（反り腰）")
                    st.write("「姿勢を良くしよう」と胸を張りすぎ・背中を反りすぎている勘違いエラーです。")
                elif body_forward > threshold:
                    st.warning("🟡 脱力過多：肩が腰より前に出ています（猫背）")
                    st.write("骨盤が後ろに倒れ、背中が丸まっています。呼吸が浅くなりやすい状態です。")
                else:
                    st.success("✨ 正常：ニュートラルでリラックスした状態です")
                    st.write("無駄な筋肉の力みを使わず、骨格だけで効率よく身体を支えられています。")
                    
        else:
            st.warning("画像が取得できませんでした。カメラに全身が映っていることを確認してください。")
