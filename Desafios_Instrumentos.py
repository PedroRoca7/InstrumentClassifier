import sys
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QGridLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.io import wavfile
import numpy as np
import librosa
import librosa.display
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import IPython.display as ipd
from IPython.display import display
import sounddevice as sd
import soundfile as sf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2  # Importando a regularização L2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.layers import Conv1D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
# Carregar os dados
df = pd.read_csv("dados/Metadata_Train.csv")
df2 = pd.read_csv("dados/Metadata_Test.csv")

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IA para Classificação de Instrumentos")
        self.setGeometry(100, 100, 600, 400)  # Aumente a largura para acomodar mais botões

        # Criar um layout vertical
        layout = QVBoxLayout()

        # Criar um layout horizontal para os botões principais
        button_layout = QVBoxLayout()

        # Botão para mostrar gráfico de treino
        btn_treino = QPushButton("Mostrar Dados de Treino")
        btn_treino.setFixedSize(200, 30)
        btn_treino.clicked.connect(self.mostrar_grafico_treino)
        button_layout.addWidget(btn_treino)

        # Botão para mostrar gráfico de teste
        btn_teste = QPushButton("Mostrar Dados de Teste")
        btn_teste.setFixedSize(200, 30)
        btn_teste.clicked.connect(self.mostrar_grafico_teste)
        button_layout.addWidget(btn_teste)

        # Botão para mostrar gráfico de acurácia e perda
        btn_acuracia_perda = QPushButton("Mostrar Acurácia e Perda")
        btn_acuracia_perda.setFixedSize(200, 30)
        btn_acuracia_perda.clicked.connect(self.plot_accuracy_loss)
        button_layout.addWidget(btn_acuracia_perda)

        btn_matriz_confusao = QPushButton("Mostrar Matriz de Confusão")
        btn_matriz_confusao.setFixedSize(200, 30)
        btn_matriz_confusao.clicked.connect(self.plot_confusion_matrix)
        button_layout.addWidget(btn_matriz_confusao)

        # Centralizar o layout dos botões principais
        button_layout.setAlignment(Qt.AlignCenter)  # Centraliza os botões

        # Adicionar o layout de botões principais ao layout vertical
        layout.addLayout(button_layout)

        # Criar um widget central e definir o layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Inicializa os atributos history, model, X_test, y_test e LE
        self.history = None  
        self.model = None  # Inicializa o atributo model
        self.X_test = None  # Inicializa o atributo X_test
        self.y_test = None  # Inicializa o atributo y_test
        self.LE = LabelEncoder()  # Inicializa o LabelEncoder como um atributo
        self.scaler = None  # Inicializa o scaler

        # Chame a função de treinamento do modelo 
        self.train_model()  

        # Criar botões para os arquivos de áudio na pasta Test_submission
        self.create_audio_buttons(layout)

    def mostrar_grafico_treino(self):
        plt.figure(figsize=(11, 5))
        sns.countplot(data=df, x='Class', hue='Class', palette='viridis', legend=False)
        plt.xlabel('Classe do Instrumento')
        plt.ylabel('Quantidade')
        plt.xticks(rotation=0)
        plt.title('Dados de Treino')
        plt.show()

    def mostrar_grafico_teste(self):
        plt.figure(figsize=(11, 5))
        sns.countplot(data=df2, x='Class', hue='Class', palette='viridis', legend=False)
        plt.xlabel('Classe do Instrumento')
        plt.ylabel('Quantidade')
        plt.xticks(rotation=0)
        plt.title('Dados de Teste')
        plt.show()
    
    def Feature_extractor(self, file):
        # Carregar o áudio com duração fixa
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast', duration=5)
        
        # 1. MFCCs com delta e delta2
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_delta = librosa.feature.delta(mfccs_features)
        mfccs_delta2 = librosa.feature.delta(mfccs_features, order=2)
        
        # 2. Características espectrais
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        
        # 3. Características rítmicas e energéticas
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        rmse = librosa.feature.rms(y=audio)
        
        # 4. Harmônicos e perceptuais
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
        
        # Combinar todas as características (média de cada característica)
        features = np.hstack([
            np.mean(mfccs_features.T, axis=0),
            np.mean(mfccs_delta.T, axis=0),
            np.mean(mfccs_delta2.T, axis=0),
            np.mean(chroma_stft.T, axis=0),
            np.mean(spectral_centroid.T, axis=0),
            np.mean(spectral_contrast.T, axis=0),
            np.mean(spectral_rolloff.T, axis=0),
            np.mean(zero_crossing_rate.T, axis=0),
            np.mean(rmse.T, axis=0),
            np.mean(tonnetz.T, axis=0)
        ])
        
        return features

    def train_model(self):
        extracted_features = []
        labels = []

        for index_num, row in tqdm(df.iterrows()):
            file_name = os.path.join(os.path.abspath('Train_submission/'), str(row["FileName"]))
            final_class_label = row["Class"]
            data = self.Feature_extractor(file_name)
            extracted_features.append([data, final_class_label])

        extracted_features_df = pd.DataFrame(extracted_features, columns=['features', 'class'])

        X = np.array(extracted_features_df['features'].tolist())
        y = np.array(extracted_features_df["class"].tolist())

        # Normalização dos dados (importante com múltiplas características)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Salvar o scaler para uso futuro em previsões
        self.scaler = scaler

        y = to_categorical(self.LE.fit_transform(y))

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Shape Of X_train:", X_train.shape)
        print("Shape Of X_test:", self.X_test.shape)
        print("Shape Of y_train:", y_train.shape)
        print("Shape Of y_test:", self.y_test.shape)    

        num_labels = y.shape[1]
        input_shape = X_train.shape[1]  # Obter dimensão correta com base nas novas características

        # Criar modelo com arquitetura melhorada
        self.model = Sequential()
        
        # Primeira camada com regularização L2
        self.model.add(Dense(128, input_shape=(input_shape,), kernel_regularizer=l2(0.001)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        
        # Segunda camada
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        
        # Terceira camada
        self.model.add(Dense(128))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        
        # Camada de saída
        self.model.add(Dense(num_labels))
        self.model.add(Activation('softmax'))

        self.model.summary()

        # Usar otimizador Adam com taxa de aprendizado personalizada
        optimizer = Adam(learning_rate=0.0005)
        self.model.compile(
            loss='categorical_crossentropy', 
            metrics=['accuracy'], 
            optimizer=optimizer
        )

        # Implementar Early Stopping e ReduceLROnPlateau
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1,
            restore_best_weights=True
        )
        
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=0.00001
        )
        
        checkpointer = ModelCheckpoint(
            filepath='saved_models/audio_classification.keras', 
            verbose=1, 
            save_best_only=True
        )
        
        # Treinar com batch_size maior e os novos callbacks
        num_epochs = 100
        num_batch_size = 64  # Aumentar o batch size

        start = datetime.now()
        self.history = self.model.fit(
            X_train, y_train, 
            batch_size=num_batch_size, 
            epochs=num_epochs, 
            validation_data=(self.X_test, self.y_test), 
            callbacks=[checkpointer, early_stopping, lr_scheduler],
            verbose=1
        )

        duration = datetime.now() - start
        print("Training Completed in time: ", duration)
        
        # Avaliação do modelo
        test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Acurácia do modelo:", test_accuracy[1])

    def plot_accuracy_loss(self):
        # Verifica se o history foi definido
        if self.history is not None:  
            fig = plt.figure(figsize=(10, 5))

            # Plot accuracy
            plt.subplot(221)
            plt.plot(self.history.history['accuracy'], 'bo--', label="accuracy")
            plt.plot(self.history.history['val_accuracy'], 'ro--', label="val_accuracy")
            plt.title("train_accuracy vs val_accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epochs")
            plt.legend()

            # Plot loss function
            plt.subplot(222)
            plt.plot(self.history.history['loss'], 'bo--', label="loss")
            plt.plot(self.history.history['val_loss'], 'ro--', label="val_loss")
            plt.title("train_loss vs val_loss")
            plt.ylabel("loss")
            plt.xlabel("epochs")

            plt.legend()
            plt.show()

    def create_audio_buttons(self, layout):
        # Criar um layout de grade para os botões de áudio
        audio_button_layout = QGridLayout()

        # Listar arquivos na pasta Test_submission
        audio_files = os.listdir("Test_submission")
        row = 0
        col = 0
        for audio_file in audio_files:
            if audio_file.endswith('.wav'):  # Verifica se é um arquivo .wav
                btn_audio = QPushButton(audio_file)
                btn_audio.setFixedSize(150, 30)  # Ajuste o tamanho do botão
                btn_audio.clicked.connect(lambda checked, filename=audio_file: self.play_audio("Test_submission/" + filename))
                audio_button_layout.addWidget(btn_audio, row, col)  # Adiciona o botão na grade

                col += 1  # Move para a próxima coluna
                if col >= 6:  # Se já tiver 6 botões na linha, vai para a próxima linha
                    col = 0
                    row += 1

        # Adicionar o layout de botões de áudio ao layout principal
        layout.addLayout(audio_button_layout)

    def play_audio(self, filename):
        # Carregar o áudio completo
        data, sample_rate = sf.read(filename)

        # Limitar a reprodução a 8 segundos
        duration = 8  # duração em segundos
        samples = int(duration * sample_rate)  # calcular o número de amostras

        # Reproduzir apenas os primeiros 8 segundos
        sd.play(data[:samples], sample_rate)  # Toca apenas os primeiros 8 segundos
        sd.wait()  # Espera até que o áudio termine de tocar

        self.predict_audio(filename)

    def predict_audio(self, filename):
        # Extrair características usando o método atualizado
        features = self.Feature_extractor(filename)
        
        # Normalizar as características usando o mesmo scaler
        features_normalized = self.scaler.transform(features.reshape(1, -1))
        
        # Realizar a previsão
        predicted_label = np.argmax(self.model.predict(features_normalized), axis=-1)
        prediction_class = self.LE.inverse_transform(predicted_label)
        
        print(f"Instrumento Previsto: {prediction_class[0]}")
        # Exibir também as probabilidades
        probabilities = self.model.predict(features_normalized)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_classes = self.LE.inverse_transform(top_3_indices)
        top_3_probs = probabilities[top_3_indices]
        
        print("Top 3 previsões:")
        for cls, prob in zip(top_3_classes, top_3_probs):
            print(f"{cls}: {prob*100:.2f}%")

    def plot_confusion_matrix(self):
        if self.model is None or self.X_test is None or self.y_test is None:
            print("Modelo ainda não foi treinado.")
            return

 
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=self.LE.classes_,
                    yticklabels=self.LE.classes_,
                    cmap='Blues')
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = App()
    mainWin.show()
    sys.exit(app.exec_())