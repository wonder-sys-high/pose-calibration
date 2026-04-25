import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import time
import queue
import io

from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# 画像と「診断結果のテキスト」をセットで保存する箱に変更
@st.cache_resource
def get_frame_queue():
    return queue.Queue(maxsize=1)

frame_queue = get_frame_queue()

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.title("姿勢キャリブレーション")
st.write("横を向いて、「一番良いと思う姿勢」を作ってください。")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose.process(img_rgb)
    status_text = "Good Posture" # デフォルト値

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

    # imgだけでなく、status_text（診断結果）も一緒に箱に入れる
    if not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put((img, status_text))

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="pose-estimation", 
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if webrtc_ctx.state.playing:
    st.markdown("---")
    if st.button("📸 30秒後に撮影して診断する"):
        countdown_placeholder = st.empty()
        
        for i in range(30, 0, -1):
            countdown_placeholder.markdown(f"### 撮影まであと **{i}** 秒...")
            time.sleep(1)
            
        countdown_placeholder.empty() # カウントダウンの文字を消す
        
        if not frame_queue.empty():
            # 画像と診断結果を取り出す
            snapshot, final_status = frame_queue.get()
            
            st.markdown("## 📊 あなたの診断結果")
            
            # --- 診断結果に応じたフィードバックの出し分け ---
            if "Forward Head" in final_status:
                st.error("⚠️ 首が前に出ています（ストレートネック予備軍）")
                st.write("**【主観と客観のズレ】**\n自分では背筋を伸ばしているつもりでも、首だけでバランスを取ろうとしています。")
                st.info("💡 **1分リセットアクション:**\n後頭部を後ろの見えない壁に押し付けるように、ゆっくり顎を引いて5秒キープを3回繰り返しましょう。")
                
            elif "Over-Tension" in final_status:
                st.error("⚠️ 腰や背中に過剰な力みがあります（反り腰・過緊張）")
                st.write("**【主観と客観のズレ】**\n「胸を張る＝良い姿勢」という勘違いにより、腰の筋肉を過剰に使って身体を支えています。")
                st.info("💡 **1分リセットアクション:**\n一度限界まで背中を丸めて脱力してください。そこから、お尻の骨（坐骨）に均等に体重が乗るポイントをミリ単位で探りましょう。")
                
            else:
                st.success("✨ 素晴らしい！ニュートラルな姿勢です")
                st.write("**【主観と客観が一致】**\n骨格で正しく身体を支えられており、無駄な力みがない理想的な状態です。")
                st.info("💡 **アクション:**\n今の「どこにも力が入っていない感覚」を脳にしっかり記憶させてください！")

            st.markdown("---")
            # ダウンロードボタン
            is_success, buffer = cv2.imencode(".jpg", snapshot)
            if is_success:
                io_buf = io.BytesIO(buffer)
                st.download_button(
                    label="📥 証拠画像（客観データ）を保存する",
                    data=io_buf.getvalue(),
                    file_name="pose_calibration.jpg",
                    mime="image/jpeg",
                    type="primary"
                )
        else:
            st.warning("画像が取得できませんでした。カメラに映った状態でお試しください。")
