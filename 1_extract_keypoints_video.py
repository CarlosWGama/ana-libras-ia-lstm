import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- CONFIGURAÇÕES ---
DATA_PATH = "./dataset"
SEQUENCE_LENGTH = 30
OUTPUT_FOLDER = "./processed_data"

# Caminhos dos modelos (Certifique-se que os arquivos .task estão na pasta)
POSE_MODEL_PATH = 'pose_landmarker_full.task'
HAND_MODEL_PATH = 'hand_landmarker.task'

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

def extract_keypoints(pose_result, hand_result):
    """
    Combina resultados de Pose e Hands em um único vetor.
    Mantém o formato original: Pose(33*4) + LE(21*3) + LD(21*3)
    """
    
    # 1. Extração de Pose (33 pontos * 4: x, y, z, visibility)
    if pose_result.pose_landmarks:
        # Pega a primeira pessoa detectada [0]
        pose_np = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_result.pose_landmarks[0]]).flatten()
    else:
        pose_np = np.zeros(33 * 4)

    # 2. Extração de Mãos (Requer mapeamento Esquerda/Direita)
    lh_np = np.zeros(21 * 3)
    rh_np = np.zeros(21 * 3)

    if hand_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            # A nova API retorna 'Left' ou 'Right'
            handedness = hand_result.handedness[idx][0].category_name
            
            # Extrai coordenadas (x, y, z)
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            
            if handedness == 'Left':
                lh_np = points
            elif handedness == 'Right':
                rh_np = points

    return np.concatenate([pose_np, lh_np, rh_np])

def process_dataset():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    actions = np.array([f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))])
    label_map = {label:num for num, label in enumerate(actions)}
    
    sequences, labels = [], []

  

    for action in actions:
        folder_path = os.path.join(DATA_PATH, action)
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"Processando classe: '{action}' ({len(video_files)} vídeos)...")

        for video_name in video_files:
            # Inicializa os DOIS detectores
            with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
                 HandLandmarker.create_from_options(hand_options) as hand_landmarker:
                video_path = os.path.join(folder_path, video_name)
                cap = cv2.VideoCapture(video_path)
                
                window = [] 
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    
                    # Conversão para mp.Image (Exigido pela nova API)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Timestamp em ms (Necessário para modo VIDEO)
                    # Usamos a propriedade do vídeo para ser mais preciso que time.time()
                    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    
                    print("AAAAAAAAAAAAAAAAAAAAAAAA", video_name, timestamp_ms)

                    # Executa as detecções
                    pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                    hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                    
                    print("BBBBBBBBBBBBBBBBBBBBBBB", video_name, timestamp_ms)
                    
                    # Extração combinada
                    keypoints = extract_keypoints(pose_results, hand_results)
                    window.append(keypoints)

                    if len(window) == SEQUENCE_LENGTH:
                        break
                
                cap.release()

                # Padding / Saving Logic (Inalterado)
                if len(window) > 0:
                    feature_size = window[0].shape[0]
                    while len(window) < SEQUENCE_LENGTH:
                        window.append(np.zeros(feature_size))
                    
                    sequences.append(np.array(window))
                    labels.append(label_map[action])

                
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    np.save(os.path.join(OUTPUT_FOLDER, 'X_data.npy'), X)
    np.save(os.path.join(OUTPUT_FOLDER, 'y_data.npy'), y)
    np.save(os.path.join(OUTPUT_FOLDER, 'labels_map.npy'), actions)

    print(f"\nSucesso! Dados salvos em {OUTPUT_FOLDER}")
    print(f"X shape: {X.shape}")

if __name__ == "__main__":
    process_dataset()