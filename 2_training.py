import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- CONFIGURAÇÕES ---
DATA_FOLDER = "./processed_data"
MODEL_NAME = "libras_model.h5"
LOG_DIR = os.path.join('Logs')

# 1. Carregar Dados
print("Carregando dados...")
X = np.load(os.path.join(DATA_FOLDER, 'X_data.npy'))
y = np.load(os.path.join(DATA_FOLDER, 'y_data.npy'))
actions = np.load(os.path.join(DATA_FOLDER, 'labels_map.npy'))

print(f"Dados carregados: {X.shape[0]} amostras")
print(f"Classes: {actions}")

# 2. Divisão Treino e Teste (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definição automática dos shapes
input_shape = (X.shape[1], X.shape[2]) # (30 frames, 1662 features)
num_classes = y.shape[1]

# 3. Arquitetura do Modelo
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu')) # Última LSTM precisa ser False

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 4. Callbacks (Essencial para um bom treino)
callbacks = [
    # Para o treino se não melhorar a validação por 20 épocas (evita overfitting)
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    
    # Salva apenas o melhor modelo durante o processo
    ModelCheckpoint(filepath=MODEL_NAME, monitor='val_categorical_accuracy', save_best_only=True, mode='max'),
    
    # Logs para visualizar no TensorBoard (opcional)
    TensorBoard(log_dir=LOG_DIR)
]

# 5. Treinamento
print("\nIniciando treinamento...")
history = model.fit(X_train, y_train, 
                    epochs=200, 
                    batch_size=32, 
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)

# 6. Visualização Rápida dos Resultados
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Train Acc')
plt.plot(history.history['val_categorical_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.show()

print(f"Modelo salvo como {MODEL_NAME}")