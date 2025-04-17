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
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import IPython.display as ipd
from IPython.display import display
import sounddevice as sd
import soundfile as sf

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
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features

    def train_model(self):
        extracted_features = []

        for index_num, row in tqdm(df.iterrows()):
            file_name = os.path.join(os.path.abspath('Train_submission/'), str(row["FileName"]))
            final_class_label = row["Class"]
            data = self.Feature_extractor(file_name)
            extracted_features.append([data, final_class_label])

        extracted_features_df = pd.DataFrame(extracted_features, columns=['features', 'class'])

        X = np.array(extracted_features_df['features'].tolist())
        y = np.array(extracted_features_df["class"].tolist())

        y = to_categorical(self.LE.fit_transform(y))  # Usando self.LE

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Shape Of X_train:", X_train.shape)
        print("Shape Of X_test:", self.X_test.shape)
        print("Shape Of y_train:", y_train.shape)
        print("Shape Of y_test:", self.y_test.shape)    

        num_labels = y.shape[1]

        self.model = Sequential()  # Armazena o modelo como um atributo da classe
        self.model.add(Dense(100, input_shape=(40,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(200))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(200))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(100))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(num_labels))
        self.model.add(Activation('softmax'))

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        num_epochs = 100
        num_batch_size = 32

        checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.keras', verbose=1, save_best_only=True)
        start = datetime.now()
        self.history = self.model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(self.X_test, self.y_test), callbacks=[checkpointer])

        duration = datetime.now() - start
        print("Training Completed in time: ", duration)

        # Avaliação do modelo
        test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)  # Usando self.model
        print("Acurácia do modelo:", test_accuracy[1])  # Imprime a acurácia

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
                if col >= 3:  # Se já tiver 3 botões na linha, vai para a próxima linha
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
        audio,sample_rate=librosa.load(filename,res_type='kaiser_fast')
        mfccs_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
        mfccs_scaled_features =np.mean(mfccs_features.T,axis=0)
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

        predicted_label = np.argmax(self.model.predict(mfccs_scaled_features), axis=-1)
        prediction_class=self.LE.inverse_transform(predicted_label)
        print(prediction_class)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = App()
    mainWin.show()
    sys.exit(app.exec_())