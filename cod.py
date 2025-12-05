import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pywt

from scipy import signal
from scipy.signal import butter, sosfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from tpot import TPOTClassifier

bckg1 = pd.read_csv("/Users/metamorphosus/Desktop/TecMonterrey/QuintoSemestre/UNAM/epilepsia/seiz/seiz_0001.csv")

# print(bckg1.describe())


def graph(df):
    tiempo = df.iloc[:, 0]
    canales = df.columns[1:]

    OFFSET = 150

    plt.figure(figsize=(16, 10))

    lista_ticks_y = []    # Para guardar la posición de cada etiqueta de canal
    lista_labels_y = []   # Para guardar el nombre de cada canal

    # Itera sobre los canales
    for i, canal in enumerate(canales):
        desplazamiento_actual = -i * OFFSET

        plt.plot(tiempo, df[canal] + desplazamiento_actual, color='black', linewidth=0.8)

        lista_ticks_y.append(desplazamiento_actual)
        lista_labels_y.append(canal)


    ax = plt.gca()

    # Pone las etiquetas de los canales en lugar de números en el eje Y
    ax.set_yticks(lista_ticks_y)
    ax.set_yticklabels(lista_labels_y)

    ax.set_xlabel("Tiempo (segundos)", fontsize=12)
    ax.set_ylabel("Canales", fontsize=12)

    # Limita el eje X al tiempo del registro
    plt.xlim(tiempo.min(), tiempo.max())
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

graph(bckg1)


def hjorth_params(x, axis = -1):
    x = np.asarray(x)
    eps = 1e-10 # Evitar divisiones con cero

    # Calculate derivatives
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)

    # Calculate variance
    x_var = np.var(x, axis=axis)  # = activity
    dx_var = np.var(dx, axis=axis)
    ddx_var = np.var(ddx, axis=axis)

    # Mobility and complexity
    mob = np.sqrt(dx_var / (x_var + eps))
    com = np.sqrt(ddx_var / (dx_var + eps)) / (mob + eps)
    return mob, com

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Aplica un filtro Butterworth pasa-banda.
    """
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    # 'sos' (second-order sections) es más estable numéricamente que 'ba'
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    y = sosfilt(sos, data)
    return y

def extract_wavelet_features(signal, wavelet = 'db4', level = 5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs es una lista [cA5, cD5, cD4, cD3, cD2, cD1]

    features = []
    for c in coeffs:
        # La energía es la suma de los cuadrados de los coeficientes
        energy = np.sum(np.square(c))
        features.append(energy)
    return features

def data_process(name1, name2):
    X_features = [] # Features
    y_label = [] # Labels

    fs = 250 # Frecuencia de muestreo
    lowcut = 0.5
    highcut = 70

    for i in range(1, 501):
        archivos1 = f"/Users/metamorphosus/Desktop/TecMonterrey/QuintoSemestre/UNAM/epilepsia/{name1}/{name1}_{i:04d}.csv"

        try:
            df_bckg = pd.read_csv(archivos1)
            file_fts = [] # Lista para las características

            for canal in df_bckg.columns[1:]:
                signal_array = df_bckg[canal].to_numpy()
                signal_array_filt = butter_bandpass_filter(signal_array, lowcut, highcut, fs)

                media = signal_array_filt.mean()
                varianza = signal_array_filt.var()
                kurtosis = pd.Series(signal_array_filt).kurtosis() # kurtosis/skew necesitan un objeto Series
                asimetria = pd.Series(signal_array_filt).skew()

                mob, com = hjorth_params(signal_array_filt)
                wavelet_fts = extract_wavelet_features(signal_array_filt)


                # f = array de frecuencia, pwr = array de potencia
                f, pwr = signal.welch(signal_array_filt, fs = fs, nperseg = int(fs * 2))
                bandas = {
                    'delta': (0, 4),
                    'theta': (4, 8),
                    'alfa': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, 150)
                }

                powerbands = []
                for banda in bandas.values():
                    freq_index = np.where((f >= banda[0]) & (f <= banda[1]))[0]
                    powerband = np.mean(pwr[freq_index]) if len(freq_index) > 0 else 0
                    powerbands.append(powerband)

                file_fts.append(media)
                file_fts.append(varianza)
                file_fts.append(kurtosis)
                file_fts.append(asimetria)
                file_fts.append(mob)
                file_fts.append(com)
                file_fts.extend(wavelet_fts)
                file_fts.extend(powerbands)

        except FileNotFoundError:
            # If a file is not found, print a message and continue to the next iteration
            # instead of breaking, which would result in empty dataframes
            print(f"Warning: File not found: {archivos1}. Skipping.")
            continue # Continue to the next file instead of breaking
        except pd.errors.EmptyDataError:
            print(f"Warning: {archivos1} is empty. Skipping.")
            continue

        X_features.append(file_fts)
        y_label.append(0)


    for i in range(1, 501):
        archivo2 = f"/Users/metamorphosus/Desktop/TecMonterrey/QuintoSemestre/UNAM/epilepsia/{name2}/{name2}_{i:04d}.csv"
        try:
            df_seiz = pd.read_csv(archivo2)

            file_fts = []
            for canal in df_seiz.columns[1:]:
                signal_array = df_seiz[canal].to_numpy()
                signal_array_filt = butter_bandpass_filter(signal_array, lowcut, highcut, fs)

                media = signal_array_filt.mean()
                varianza = signal_array_filt.var()
                kurtosis = pd.Series(signal_array_filt).kurtosis()
                asimetria = pd.Series(signal_array_filt).skew()


                mob, com = hjorth_params(signal_array_filt)
                wavelet_fts = extract_wavelet_features(signal_array_filt)

                # f = array de frecuencia, pwr = array de potencia
                f, pwr = signal.welch(signal_array_filt, fs = fs, nperseg = int(fs * 2))
                bandas = {
                    'delta': (0, 4),
                    'theta': (4, 8),
                    'alfa': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, 150)
                }

                powerbands = []
                for banda in bandas.values():
                    freq_index = np.where((f >= banda[0]) & (f <= banda[1]))[0]
                    powerband = np.mean(pwr[freq_index]) if len(freq_index) > 0 else 0
                    powerbands.append(powerband)

                file_fts.append(media)
                file_fts.append(varianza)
                file_fts.append(kurtosis)
                file_fts.append(asimetria)
                file_fts.append(mob)
                file_fts.append(com)
                file_fts.extend(wavelet_fts)
                file_fts.extend(powerbands)

        except FileNotFoundError:
            print(f"Warning: File not found: {archivo2}. Skipping.")
            continue # Continue to the next file instead of breaking
        except pd.errors.EmptyDataError:
            print(f"Warning: {archivo2} is empty. Skipping.")
            continue

        X_features.append(file_fts)
        y_label.append(1)


    X = pd.DataFrame(X_features)
    y = pd.Series(y_label)

    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # X_scaled = StandardScaler().fit_transform(X)

    return X, y


def training(X_tarea, y_tarea, name1, name2, model_name='rf'):
    # Ensure that there are enough samples to split
    if len(X_tarea) == 0:
        print(f"Error: No data available for training {name1} vs {name2}.")
        return
    if len(y_tarea.unique()) < 2:
        print(f"Error: Not enough classes in y_tarea for stratified split ({name1} vs {name2}).")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X_tarea,
        y_tarea,
        test_size = 0.20,
        random_state = 88,
        stratify = y_tarea  # Asegura que haya 50% de clase 0 y 50% de clase 1 en ambos sets
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    if model_name == 'rf':
        print(f"--- Modelo: Random Forest ---")
        classifier = RandomForestClassifier(n_estimators = 100, random_state = 97, n_jobs = -1)
        # The tpot object was declared here but then used on subsequent lines. Moved to global scope
        # tpot = TPOTClassifier(
        # generations=5,
        # population_size=20,
        # scoring='f1',  # ¡La métrica que nos importa!
        # verbosity=2,   # Para que imprima su progreso
        # random_state=42,
        # n_jobs=-1
        # )
        #tpot_classifier = TPOTClassifier(
            #generations=5,
            #population_size=20,
            #scoring='f1',  # ¡La métrica que nos importa!
            #verbosity=0,   # Reduced verbosity to avoid excessive output
            #random_state=42,
            #n_jobs=-1
        #)

        #tpot_classifier.fit(X_train, y_train)
        #print(f"Mejor F1-Score en test: {tpot_classifier.score(X_test, y_test)}")
        #classifier = tpot_classifier.fitted_pipeline_ # Use the best pipeline found by TPOT

    #elif model_name == 'svm':
        #print(f"--- Modelo: Support Vector Machine (SVC) ---")
        #classifier = SVC(kernel='rbf', C=1.0, random_state = 97)

    #elif model_name == 'xgb':
        #print(f"--- Modelo: XGBoost ---")
        #classifier = XGBClassifier(n_estimators = 100,
                                   #random_state = 97,
                                   #n_jobs = -1, use_label_encoder=False, eval_metric='logloss')

    #elif model_name == 'nn':
        #print(f"--- Modelo: Red Neuronal (MLP) ---")
        # Una red simple: 2 capas ocultas con 100 y 50 neuronas
        #classifier = MLPClassifier(hidden_layer_sizes=(100, 50, 30, 20),
                                   #max_iter=500,
                                   #random_state=97, early_stopping=True)


    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    print(f'F1 score = {f1:.4f}')

    print(f'---Reporte de clasificación---')
    print(classification_report(y_test, y_pred, target_names = [f'{name1}(0)', f'{name2}(1)']))

    print(f'---Confussion Matrix---')
    print(confusion_matrix(y_test, y_pred))


#def visualizar_clusters(X, y, title):
    # This function was called but not defined. Adding a placeholder to avoid NameError
    #print(f"Visualization for '{title}' is not implemented.")
    # If PCA is intended for visualization, it can be added here
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', legend='full')
    # plt.title(title)
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.show()


X_tarea1, y_tarea1 = data_process('bckg', 'seiz')
X_tarea2, y_tarea2 = data_process('pre', 'seiz')
X_tarea3, y_tarea3 = data_process('bckg', 'pre')


modelos_a_probar = ['rf']
print(f'\n---- Inciando Tarea 1: bckg vs. seiz ----')
for model in modelos_a_probar:
    training(X_tarea1, y_tarea1, 'bckg', 'seiz', model_name=model)
#visualizar_clusters(X_tarea1, y_tarea1, 'bckg (0) vs. seiz (1)')

print(f'\n---- Inciando Tarea 2: pre vs. seiz ----')
for model in modelos_a_probar:
    training(X_tarea2, y_tarea2, 'pre', 'seiz', model_name=model)
#visualizar_clusters(X_tarea2, y_tarea2, 'pre (0) vs. seiz (1)')

print(f'\n---- Inciando Tarea 3: bckg vs. pre ----')
for model in modelos_a_probar:
    training(X_tarea3, y_tarea3, 'bckg', 'pre', model_name=model)
#visualizar_clusters(X_tarea3, y_tarea3, 'bckg (0) vs. pre (1)')