import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
import os

# --- 1. CONFIGURAÇÕES ---
# Caminho do vídeo a ser testado. 
# Deixe "livre" para buscar um vídeo do dataset
VIDEO_PATH = "video/video2.mp4" 

# Caminhos dos modelos
MODEL_PATH = "./models_saved/libras_model_test.h5"
LABELS_PATH = "./processed_data/labels_map.npy"
POSE_MODEL_PATH = 'pose_landmarker_heavy.task'
HAND_MODEL_PATH = 'hand_landmarker.task'

# Parâmetros (Devem ser IGUAIS ao training.py e extract.py)
SEQUENCE_LENGTH = 45  # O modelo espera sequências de 30 frames
THRESHOLD = 0.6       # Confiança mínima para exibir a predição

# --- 2. CARREGAR MODELO E CLASSES ---
if not os.path.exists(MODEL_PATH):
    print(f"ERRO: Modelo não encontrado em {MODEL_PATH}")
    exit()

print(f"Carregando modelo: {MODEL_PATH}...")
model = load_model(MODEL_PATH)
actions = np.load(LABELS_PATH)
print(f"Classes carregadas: {actions}")

# --- 3. CONFIGURAÇÃO MEDIAPIPE TASKS ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = vision.RunningMode

# Configuração Pose
pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

# Configuração Hands
hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5
)

# --- 4. FUNÇÃO DE EXTRAÇÃO DE KEYPOINTS ---
# Esta função deve ser EXATAMENTE a mesma usada no treinamento
def extract_keypoints(pose_result, hand_result):
    """
    Combina resultados de Pose e Hands em um vetor de 258 features.
    Pose (33*4) + ME (21*3) + MD (21*3)
    """
    # 1. Pose: 33 pontos * 4 (x, y, z, visibility)
    if pose_result.pose_landmarks:
        pose_np = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_result.pose_landmarks[0]]).flatten()
    else:
        pose_np = np.zeros(33 * 4)

    # 2. Mãos: 21 pontos * 3 (x, y, z) cada
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

# --- 5. ALGORITMO PRINCIPAL DE CAPTURA E IDENTIFICAÇÃO ---
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_source}")
        return

    # Inicializa Detectores do MediaPipe
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:

        sequence = []
        current_prediction = "..."
        current_prob = 0.0

        print("\nIniciando processamento... Pressione 'q' para sair.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # O MediaPipe requer imagem em RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Timestamp (ms) obrigatório para modo VIDEO
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # --- DETECÇÃO ---
            pose_res = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_res = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            # --- PREPARAÇÃO DOS DADOS ---
            keypoints = extract_keypoints(pose_res, hand_res)
            sequence.append(keypoints)
            
            # Mantém janela deslizante do tamanho correto
            sequence = sequence[-SEQUENCE_LENGTH:]

            # --- INFERÊNCIA ---
            if len(sequence) == SEQUENCE_LENGTH:
                # Expande dimensão para (1, 30, 258)
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                prediction_idx = np.argmax(res)
                confidence = res[prediction_idx]

                if confidence > THRESHOLD:
                    current_prediction = actions[prediction_idx]
                    current_prob = confidence
                else:
                    current_prediction = "..." # Incerteza

            # --- VISUALIZAÇÃO ---
            # Mostra o status na tela
            cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
            display_text = f"{current_prediction} ({current_prob:.2f})"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Desenhar landmarks (Opcional, apenas para debug visual)
            # Para desenhar, precisaríamos converter mp_image de volta ou usar o frame original
            # e usar mp.solutions.drawing_utils (que trabalha com estrutura da API antiga,
            # então pode ser chato converter os Landmarks da Task API para desenhar fácil).
            # Vamos focar na predição.

            cv2.imshow('Libras Recognition - Video', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# --- 6. EXECUÇÃO ---
if __name__ == "__main__":
    # Verifica se o usuário definiu um vídeo, senão tenta pegar o primeiro da pasta dataset para teste
    target_video = VIDEO_PATH
    
    if target_video == "livre":
        # Tenta achar um vídeo de teste automático
        print("Nenhum vídeo definido em VIDEO_PATH.")
        if os.path.exists("./dataset"):
             # Tenta achar um arquivo qualquer
             for root, dirs, files in os.walk("./dataset"):
                 for file in files:
                     if file.endswith((".mp4", ".avi")):
                         target_video = os.path.join(root, file)
                         print(f"Usando vídeo de exemplo encontrado: {target_video}")
                         break
                 if target_video != VIDEO_PATH: break
    

    process_video(target_video)
