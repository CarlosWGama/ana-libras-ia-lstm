import cv2
import numpy as np
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

# --- CONFIGURAÇÕES ---
MODEL_PATH = "./models_saved/libras_model_test.h5"     # Seu modelo treinado
LABELS_PATH = "./processed_data/labels_map.npy" # Arquivo com nomes das classes
POSE_MODEL_PATH = 'pose_landmarker_full.task'
HAND_MODEL_PATH = 'hand_landmarker.task'
SEQUENCE_LENGTH = 30
THRESHOLD = 0.8  # Só mostra a predição se a confiança for > 80%

# --- CARREGAR MODELO E LABELS ---
model = load_model(MODEL_PATH)
actions = np.load(LABELS_PATH)
print(f"Modelo carregado. Classes: {actions}")

# --- CONFIGURAÇÃO MEDIAPIPE TASKS ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = vision.RunningMode

# Opções Pose
PoseLandmarker = vision.PoseLandmarker
pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

# Opções Hands
HandLandmarker = vision.HandLandmarker
hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5
)

# --- FUNÇÃO AUXILIAR (MESMA DO TREINO) ---
def extract_keypoints(pose_result, hand_result):
    # 1. Pose
    if pose_result.pose_landmarks:
        pose_np = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_result.pose_landmarks[0]]).flatten()
    else:
        pose_np = np.zeros(33 * 4)

    # 2. Mãos
    lh_np = np.zeros(21 * 3)
    rh_np = np.zeros(21 * 3)

    if hand_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            
            if handedness == 'Left':
                lh_np = points
            elif handedness == 'Right':
                rh_np = points

    return np.concatenate([pose_np, lh_np, rh_np])

# --- LOOP PRINCIPAL ---
sequence = [] # Buffer para guardar os últimos 30 frames
sentence = [] # Histórico de frases
predictions = []

cap = cv2.VideoCapture(0)

# Inicializa Detectores
with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     HandLandmarker.create_from_options(hand_options) as hand_landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Espelhamento para ficar natural (opcional)
        # frame = cv2.flip(frame, 1)

        # Prepara imagem para MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)

        # Detecção
        pose_res = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_res = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Extração de Features
        keypoints = extract_keypoints(pose_res, hand_res)
        sequence.append(keypoints)
        
        # Mantém apenas os últimos 30 frames
        sequence = sequence[-SEQUENCE_LENGTH:]

        # Lógica de Predição
        if len(sequence) == SEQUENCE_LENGTH:
            # O modelo espera (1, 30, Features)
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            # Pega o índice da maior probabilidade
            predicted_action = actions[np.argmax(res)]
            confidence = res[np.argmax(res)]

            # Visualização simples na tela
            color = (0, 255, 0) if confidence > THRESHOLD else (0, 0, 255)
            
            # Barra de confiança
            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            text = f"{predicted_action} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Debug no terminal (opcional)
            # print(f"Predição: {predicted_action} | Confiança: {confidence:.2f}")

        # Mostra o vídeo
        cv2.imshow('Libras Recognition', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()