import streamlit as st
import cv2
import numpy as np
from PIL import Image

from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# MediaPipeの姿勢推定モデルを初期化（最高精度レベル1）
pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5,
    model_complexity=1
)

st.title("姿勢キャリブレーション")
st.markdown("### 主観と客観のズレを暴く詳細診断")

st.info("💡 **【撮影のコツ】**\n下のボタンを押し、「写真を撮る」を選択してください。スマホ標準カメラの**10秒タイマー**を使って、少し離れた場所にスマホを立てかけて横向きの全身を撮影するのがおすすめです。")

# 画像アップローダー（スマホのカメラ起動と連携）
uploaded_file = st.file_uploader("📸 ここをタップして撮影 / 写真を選択", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の読み込みと前処理
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # RGBA(透過)などの形式を標準のRGBに変換
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_array

    st.markdown("---")
    with st.spinner("AIが骨格レベルで詳細に解析しています..."):
        results = pose.process(img_rgb)

    if results.pose_landmarks:
        annotated_img = img_rgb.copy()
        
        # ベースの骨格線を「薄いグレー」で控えめに描画
        mp_drawing.draw_landmarks(
            annotated_img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=1)  
        )

        # 必要な関節ポイントを取得（左半身を基準）
        landmarks = results.pose_landmarks.landmark
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        h, w, _ = annotated_img.shape
        ear_x, ear_y = int(ear.x * w), int(ear.y * h)
        shoulder_x, shoulder_y = int(shoulder.x * w), int(shoulder.y * h)
        hip_x, hip_y = int(hip.x * w), int(hip.y * h)
        knee_x, knee_y = int(knee.x * w), int(knee.y * h)
        ankle_x, ankle_y = int(ankle.x * w), int(ankle.y * h)

        head_shift = ear_x - shoulder_x
        body_shift = shoulder_x - hip_x
        
        # 画面幅に応じた厳密な閾値（約4%）
        threshold = w * 0.04 

        # --- 可視化ロジック ---
        # 1. 理想の姿勢ライン（ブルー：腰を基準とした真っ直ぐな縦線）
        cv2.line(annotated_img, (hip_x, 0), (hip_x, h), (50, 150, 255), 3)
        
        # 2. あなたの実際の骨格ライン（オレンジ：全身を繋ぐ）
        cv2.line(annotated_img, (ear_x, ear_y), (shoulder_x, shoulder_y), (255, 100, 50), 4) # 首
        cv2.line(annotated_img, (shoulder_x, shoulder_y), (hip_x, hip_y), (255, 100, 50), 4) # 胴体
        cv2.line(annotated_img, (hip_x, hip_y), (knee_x, knee_y), (255, 100, 50), 4)         # 太もも
        cv2.line(annotated_img, (knee_x, knee_y), (ankle_x, ankle_y), (255, 100, 50), 4)     # すね

        # 診断結果画像の表示
        st.image(annotated_img, caption="ブルー：理想の姿勢ライン / オレンジ：あなたの実際の姿勢", use_column_width=True)

        # ユーザーに「正解」を直感的に伝える解説ブロック
        st.info("💡 **【画像の見方】青とオレンジの線がピタッと重なっていれば100点満点！**\n\n青い縦線（理想のライン）の上に、あなたの全身のオレンジ線（耳・肩・腰・膝・足首）が乗っていれば、首や腰に最も負担のかからない正しい姿勢です。")

        st.markdown("## 📊 詳細診断レポート")

        # --- 1. 首・頭の判定 ---
        st.markdown("### 1. ストレートネック度（頭の突出）")
        if head_shift > threshold * 1.5:
            st.error("⚠️ 重度：基準より頭がかなり前に出ています")
            st.write("首の筋肉だけで重い頭（約5kg）を必死に支えている状態です。慢性的な肩こりや自律神経の乱れに直結しやすいエラーです。")
        elif head_shift > threshold:
            st.warning("🟡 軽度：基準より頭が少し前に出ています")
            st.write("デスクワーク中によく見られる姿勢です。自分では真っ直ぐなつもりでも、画面に引き寄せられています。")
        else:
            st.success("✨ 正常：理想的な位置です")
            st.write("頭の重さが背骨全体に正しく分散されており、首への負担が最小限に抑えられています。")

        # --- 2. 腰・背中の判定 ---
        st.markdown("### 2. 腰の過緊張度（反り腰・力み）")
        if body_shift < -threshold:
            st.error("⚠️ 過緊張：肩が腰より極端に後ろにあります")
            st.write("「姿勢を良くしよう」と意識するあまり、胸を張りすぎ・背中を反りすぎている典型的な勘違いエラーです。腰痛の大きな原因になります。")
        elif body_shift > threshold:
            st.warning("🟡 脱力過多：肩が腰より前に出ています（猫背）")
            st.write("骨盤が後ろに倒れ、背中が丸まっています。呼吸が浅くなり、集中力が低下しやすい状態です。")
        else:
            st.success("✨ 正常：ニュートラルでリラックスした状態です")
            st.write("無駄な筋肉の力みを使わず、骨格（骨組み）だけで効率よく身体を支えられています。")

    else:
        st.error("全身の骨格がうまく検出できませんでした。もう少し離れて、明るい場所で再度撮影してみてください。")
