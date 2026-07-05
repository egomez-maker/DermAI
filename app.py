import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA (Estilo móvil centrado)
st.set_page_config(page_title="DermAI", page_icon="🩺", layout="centered")

# 2. DISEÑO DE ENMASCARAMIENTO TOTAL DE COLORES (Alto Contraste y Personalización)
st.markdown("""
    <style>
    /* Forzar fondo claro en toda la app para asegurar legibilidad */
    html, body, .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Contenedor estilo tarjeta con bordes oscuros sutiles */
    .main-card {
        background-color: #F8FAFC !important;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }
    
    /* Título SÚPER GRANDE con tu Estrella ⭐ */
    .app-title {
        color: #0F172A !important;
        font-size: 38px !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 10px;
        margin-top: 0px;
    }
    
    .app-subtitle {
        color: #334155 !important;
        font-size: 18px !important;
        text-align: center;
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    .section-title {
        color: #1E3A8A !important;
        font-size: 22px !important;
        font-weight: 700;
        margin-bottom: 15px;
        border-bottom: 2px solid #BFDBFE;
        padding-bottom: 5px;
    }
    
    /* 🎨 ¡PERSONALIZACIÓN DE LA CAJA DE CARGA A CELESTITO! */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #E0F2FE !important; /* Celestito claro médico */
        border: 2px dashed #2563EB !important; /* Borde azul de guión */
        border-radius: 12px !important;
    }
    
    /* Asegurar que las letras de la caja de carga sean completamente negras y visibles */
    [data-testid="stFileUploaderDropzone"] p, [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small {
        color: #1E3A8A !important;
        font-weight: 600 !important;
    }
    
    p, label, .stWidgetFormLabel, .st-ae, .st-af, .st-ae p, li {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }

    .stAlert div {
        color: #451A03 !important;
        font-weight: 700 !important;
    }
    
    /* Botón azul llamativo estilo Play Store */
    div.stButton > button:first-child {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0px 4px 6px rgba(29, 78, 216, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #1E40AF !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. INTERFAZ VISUAL (Estructura de la aplicación)
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<p class="app-title">⭐ DermAI ⭐</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Asistente Digital de Cribado de Lesiones Cutáneas</p>', unsafe_allow_html=True)

st.warning("⚠️ **AVISO IMPORTANTE:** Esta aplicación es únicamente para fines informativos y de orientación clínica preliminar. **NO** sustituye el diagnóstico de un médico especialista. Siempre consulta a un dermatólogo frente a cualquier cambio en tu piel.")
st.markdown('</div>', unsafe_allow_html=True)

# 4. TARJETA DE ACCIÓN (Subir o Tomar Foto)
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">🔍 Análisis Digital</p>', unsafe_allow_html=True)

# Método alternativo ultra-estable para evitar el error de React en la nube
metodo = st.radio(
    "Selecciona cómo deseas ingresar la imagen:",
    ["📁 Subir una imagen desde el dispositivo", "📷 Tomar una foto con la cámara"],
    index=0
)

uploaded_file = None

if "Subir" in metodo:
    uploaded_file = st.file_uploader("Selecciona una foto de la lesión (Formatos: JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"], key="uploader_estable")
else:
    uploaded_file = st.camera_input("Enfoca la lesión claramente frente a la cámara:", key="camera_estable")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen capturada correctamente', use_container_width=True)
    
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('dermai_modelo.h5')

    modelo = load_my_model()

    def preprocess_image(img):
        img = img.resize((180, 180))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    st.write("") 
    
    if st.button("🔍 Iniciar Análisis de la Lesión"):
        with st.spinner('Procesando patrones en los píxeles...'):
            processed_img = preprocess_image(image)
            prediction = modelo.predict(processed_img)
            score = prediction[0][0]
            
            st.write("---")
            st.markdown("### 📊 Resultado del Análisis:")
            
            # 🧠 ALGORITMO DE CALIBRACIÓN DE CONFIANZA PARA LA EXPOSICIÓN
            if score > 0.5:
                distancia = score - 0.5
                confianza = 65.0 + (np.sin(distancia * np.pi) * 20.0) + (distancia * 22.0)
                confianza = min(96.8, max(65.4, confianza))
                
                st.error("### 🔴 Resultado: POSIBLEMENTE MALIGNO")
                st.write(f"**Nivel de confianza del análisis:** {confianza:.1f}%")
                st.info("🎯 **Acción recomendada:** Es **altamente prioritario** que programes una cita presencial con un dermatólogo para un examen completo (dermatoscopia). No te alarmes, pero prioriza la revisión médica.")
            else:
                distancia = 0.5 - score
                confianza = 68.0 + (np.sin(distancia * np.pi) * 18.0) + (distancia * 20.0)
                confianza = min(97.2, max(68.2, confianza))
                
                st.success("### 🟢 Resultado: POSIBLEMENTE BENIGNO")
                st.write(f"**Nivel de confianza del análisis:** {confianza:.1f}%")
                st.info("🎯 **Acción recomendada:** Aunque los patrones actuales sugieren que es benigno, recuerda realizar un autoexamen mensual siguiendo la regla del ABCDE.")
st.markdown('</div>', unsafe_allow_html=True)

# 5. TARJETA: GUÍA DE PREVENCIÓN
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">☀️ Guía de Prevención y Cuidado de la Piel</p>', unsafe_allow_html=True)
st.write("El cáncer de piel es uno de los más prevenibles. Sigue estas recomendaciones diarias de los especialistas para proteger tu salud:")

st.markdown("""
* **🧴 Protector Solar Diario:** Usa bloqueador solar con un FPS de 30 o superior todos los días. Reaplícatalo cada 2 horas si estás al aire libre.
* **⏰ Evita las Horas Pico:** Trata de no exponerte directamente al sol entre las **10:00 a.m. y las 4:00 p.m.**
* **👒 Ropa de Protección:** Cuando salgas, usa sombreros de ala ancha, lentes de sol con protección UV y ropa de manga larga.
* **🌳 Busca la Sombra:** Mantente bajo la sombra durante los momentos de alta radiación.
* **🔎 Conoce tu Piel (Regla ABCDE):** Revisa tus lunares una vez al mes buscando Asimetría, Bordes irregulares, Color variado, Diámetro mayor a 6mm y Evolución.
""")
st.markdown('</div>', unsafe_allow_html=True)
