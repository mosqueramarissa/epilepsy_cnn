# Detección y Predicción de Epilepsia con Deep Learning (CNN-STFT)
Sistema de identificación de crisis epilépticas (Ictal) y predicción temprana (Pre-ictal) utilizando análisis espectral y redes neuronales convolucionales sobre señales EEG multicanal.

# Tabla de Contenidos
1. Descripción del proyecto
2. Arquitectura del modelo
3. Preprocesamiento de datos

# Descripción del proyecto
A diferencia de los métodos clásicos que dependen de la extracción manual de características, este enfoque utiliza Deep Learning para aprender patrones complejos directamente de la representación Tiempo-Frecuencia de la señal.

El sistema aborda dos tareas críticas:

1. Identificación: Clasificación entre estado Normal (bckg) y Ataque (seiz).
2. Predicción: Clasificación entre estado Normal (bckg) y Previo al Ataque (pre).


# Arquitectura del modelo
El núcleo del sistema es una Red Neuronal Convolucional (CNN-2D) diseñada para procesar imágenes espectrales.
- Entrada: Espectrogramas generados vía STFT (Short-Time Fourier Transform). Esto convierte la señal 1D en una imagen 2D de Tiempo vs. Frecuencia.

- Red: Arquitectura secuencial de 3 bloques convolucionales (Conv2d -> BatchNorm -> ReLU -> MaxPool) seguidos de un Global Average Pooling y un clasificador lineal.

- Estrategia: Fusión de sensores (promedio global de 19 canales) generar una representación robusta de la actividad cerebral global

# Dataset y Preprocesamiento
Los datos consisten en registros EEG de 19 canales (Sistema 10-20 Internacional).

Pipeline de Procesamiento:
1. Carga: Lectura de archivos CSV (15361 muestras 250 Hz).
2. Fusión: Promedio de los 19 canales para reducir dimensionalidad y ruido local.
3. Filtrado: Filtro Butterworth Pasa-Banda de 5to orden (0.5 Hz - 60 Hz) para eliminar drift DC y ruido muscular de alta frecuencia.
4. Transformación: Generación de STFT (Ventana Hann, 1 seg) y conversión a escala Logarítmica (dB).
5. Normalización: Escalado Min-Max por imagen.
