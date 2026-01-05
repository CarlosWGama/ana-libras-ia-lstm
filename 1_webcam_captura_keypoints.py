import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    # Exemplo extraindo pose e mãos. Se não detectar, preenche com zeros.
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatena tudo em um único vetor de features
    return np.concatenate([pose, lh, rh])

# Loop principal de captura (simplificado)
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Detecção e Extração
        image, results = mediapipe_detection(frame, holistic) # Função auxiliar de detecção
        keypoints = extract_keypoints(results)
        
        # Aqui você acumularia 'keypoints' em uma lista até ter 30 frames (uma sequência)
        # sequence.append(keypoints)