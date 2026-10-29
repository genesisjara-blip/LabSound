import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
import scipy.fft as fft
import plotly.graph_objects as go

# Configuración de la aplicación
st.set_page_config(page_title="LabSound", page_icon="🎵", layout="wide")

st.title("🎵 LabSound - Laboratorio Virtual de Acústica")
st.subheader("Análisis Espectral y Acústico de Ondas Musicales")

# --- BARRA LATERAL (CONFIGURACIÓN) ---
st.sidebar.header("⚙️ Configuración del Ensayo")

# Opción 1: Selector de notas musicales estándar
notas_dict = {
    "Do4 (C4)": 261.63,
    "Re4 (D4)": 293.66,
    "Mi4 (E4)": 329.63,
    "Fa4 (F4)": 349.23,
    "Sol4 (G4)": 392.00,
    "La4 (A4)": 440.00,
    "Si4 (B4)": 493.88,
    "Personalizada": 0.0
}

modo_nota = st.sidebar.selectbox("Selecciona la nota o frecuencia teórica:", list(notas_dict.keys()))

if modo_nota == "Personalizada":
    frecuencia_teorica = st.sidebar.number_input("Ingresa frecuencia teórica (Hz):", min_value=10.0, max_value=20000.0, value=440.0, step=1.0)
else:
    frecuencia_teorica = notas_dict[modo_nota]

st.sidebar.info(f"**Frecuencia Teórica Patrón:** {frecuencia_teorica:.2f} Hz")

# --- PANTALLA PRINCIPAL ---
st.write("### 🎙️ Captura de Audio Real")
st.write("Haz sonar el teclado virtual o generador de tonos y graba el audio a través del micrófono:")

# Captura de audio usando el sensor del navegador/dispositivo
audio_value = st.audio_input("Presiona para Grabar / Detener")

if audio_value is not None:
    # 1. Leer el archivo de audio grabado por el micrófono
    sample_rate, audio_data = wav.read(audio_value)
    
    # Si el audio es estéreo, tomar un solo canal (mono)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
        
    # Normalizar la señal
    N = len(audio_data)
    
    if N > 0:
        # 2. Procesamiento matemático con Transformada Rápida de Fourier (FFT)
        yf = fft.rfft(audio_data)
        xf = fft.rfftfreq(N, 1 / sample_rate)
        amplitud = np.abs(yf)

        # Buscar el pico de frecuencia fundamental (omitir componente DC/0 Hz)
        idx_pico = np.argmax(amplitud[1:]) + 1
        frecuencia_medida = xf[idx_pico]

        # 3. Cálculo del Error Relativo Porcentual
        error_relativo = abs(frecuencia_medida - frecuencia_teorica) / frecuencia_teorica * 100

        # 4. Despliegue de Métricas en Pantalla
        col1, col2, col3 = st.columns(3)
        col1.metric("Frecuencia Teórica", f"{frecuencia_teorica:.2f} Hz")
        col2.metric("Frecuencia Medida", f"{frecuencia_medida:.2f} Hz")
        col3.metric("Error Relativo Porcentual", f"{error_relativo:.2f} %")

        # 5. Gráfico interactivo con Plotly (hasta 2000 Hz)
        limite_grafico = np.where(xf <= 2000)[0][-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xf[:limite_grafico], y=amplitud[:limite_grafico], mode='lines', name='Espectro FFT', line=dict(color='#00CC96')))
        
        # Marcar el pico detectado
        fig.add_vline(x=frecuencia_medida, line_dash="dash", line_color="red", annotation_text=f"Pico: {frecuencia_medida:.1f} Hz")

        fig.update_layout(
            title="Espectro de Frecuencia en Tiempo Real (FFT)",
            xaxis_title="Frecuencia (Hz)",
            yaxis_title="Amplitud / Potencia",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Por favor, presiona el botón de grabación para permitir acceso al micrófono y tomar la muestra.")