import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp

# MediaPipeの姿勢推定モデルを初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

st.title("姿勢キャリブレーション")
st.write("横を向いて、「一番良いと思う姿勢」を作ってください。")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # 関節の点と線を引く（薄く表示）
        mp_drawing.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
        )

        landmarks = results.pose_landmarks.landmark
        
        # 画面に大きく映る側のポイントを取得（ここでは簡易的に左側を使用）
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        # 画像の縦横のサイズを取得し、ピクセル座標に変換
        h, w, _ = img.shape
        ear_x, ear_y = int(ear.x * w), int(ear.y * h)
        shoulder_x, shoulder_y = int(shoulder.x * w), int(shoulder.y * h)
        hip_x, hip_y = int(hip.x * w), int(hip.y * h)

        # 判定ロジック：X座標（横のズレ）を計算
        head_shift = ear_x - shoulder_x
        body_shift = shoulder_x - hip_x

        status_text = "Good Posture (ニュートラル)"
        color = (0, 255, 0) # 緑色

        # エラー判定の厳しさ（画面幅の約5%を許容範囲とする）
        threshold = w * 0.05 

        # 耳が肩より前ならストレートネック、肩が腰より前なら猫背/後ろなら反り腰過緊張
        if head_shift > threshold:
            status_text = "Error: Forward Head (首の力み)"
            color = (0, 0, 255) # 赤色
        elif abs(body_shift) > threshold:
            status_text = "Error: Over-Tension (反り腰・背中の力み)"
            color = (0, 0, 255)

        # 視覚的フィードバック：腰を基準とした「理想の垂直線（水色）」を描画
        cv2.line(img, (hip_x, 0), (hip_x, h), (255, 255, 0), 2)
        
        # 判定テキストを画面に描画
        cv2.putText(img, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="pose-estimation", 
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
