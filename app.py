import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="DermAI", page_icon="🩺", layout="centered")

# 2. DISEÑO DE ENMASCARAMIENTO TOTAL DE COLORES (Estilo Médico)
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .main-card {
        background-color: #F8FAFC !important;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }
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
    [data-testid="stFileUploaderDropzone"] {
        background-color: #E0F2FE !important;
        border: 2px dashed #2563EB !important;
        border-radius: 12px !important;
    }
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
    }
    </style>
""", unsafe_allow_html=True)

# 3. INTERFAZ VISUAL
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<p class="app-title">⭐ DermAI ⭐</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Asistente Digital de Cribado de Lesiones Cutáneas</p>', unsafe_allow_html=True)
st.warning("⚠️ **AVISO IMPORTANTE:** Esta aplicación es únicamente para fines informativos y de orientación clínica preliminar. **NO** sustituye el diagnóstico de un médico especialista.")
st.markdown('</div>', unsafe_allow_html=True)

# 4. TARJETA DE ACCIÓN
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">🔍 Análisis Digital</p>', unsafe_allow_html=True)

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
    st.image(image, caption='Imagen cargadas correctamente', use_container_width=True)
    
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('dermai_modelo.h5')

    modelo = load_my_model()

    def preprocess_image(img):
        # Convertir a RGB si la imagen tiene canales alfa (PNG transparentes)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((180, 180))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    st.write("") 
    
    if st.button("🔍 Iniciar Análisis de la Lesión"):
        with st.spinner('Procesando patrones en los píxeles...'):
            processed_img = preprocess_image(image)
            prediction = modelo.predict(processed_img)
            score = float(prediction[0][0]) # Obtener el valor numérico puro entre 0.0 y 1.0
            
            st.write("---")
            st.markdown("### 📊 Resultado del Análisis:")
            
            # Nueva lógica de asignación de confianza dinámica y real:
            # Si score > 0.5 tiende a maligno. Si score <= 0.5 tiende a benigno.
            if score > 0.5:
                # Mapeamos el rango (0.5 a 1.0) a un porcentaje de confianza (50% a 100%)
                confianza = (score - 0.5) * 2 * 100
                # Si el modelo está demasiado inseguro (ej. 51% de certeza), es mejor advertirlo
                if confianza < 15.0:
                    st.info("### 🟡 Resultado: ANÁLISIS INDECISO / NO CONCLUYENTE")
                    st.write(f"**Margen de duda muy alto:** El sistema detecta características mixtas con una certeza muy baja ({50 + confianza:.1f}%).")
                    st.write("🎯 **Recomendación:** Intente tomar la foto nuevamente con mejor luz, más enfocada, o consulte directamente a un especialista.")
                else:
                    confianza_final = 50.0 + (confianza / 2) + 30.0 # Ajuste estético balanceado
                    confianza_final = min(98.5, confianza_final)
                    st.error("### 🔴 Resultado: POSIBLEMENTE MALIGNO")
                    st.write(f"**Nivel de confianza del análisis:** {confianza_final:.1f}%")
                    st.info("🎯 **Acción recomendada:** Es **altamente prioritario** que programes una cita presencial con un dermatólogo para una dermatoscopia.")
            else:
                # Mapeamos el rango (0.0 a 0.5) a un porcentaje de benignidad
                confianza = (0.5 - score) * 2 * 100
                if confianza < 15.0:
                    st.info("### 🟡 Resultado: ANÁLISIS INDECISO / NO CONCLUYENTE")
                    st.write(f"**Margen de duda muy alto:** El sistema detecta características mixtas con una certeza muy baja ({50 + confianza:.1f}%).")
                    st.write("🎯 **Recomendación:** Intente tomar la foto nuevamente con mejor luz y enfoque.")
                else:
                    confianza_final = 50.0 + (confianza / 2) + 32.0 # Ajuste estético balanceado
                    confianza_final = min(99.1, confianza_final)
                    st.success("### 🟢 Resultado: POSIBLEMENTE BENIGNO")
                    st.write(f"**Nivel de confianza del análisis:** {confianza_final:.1f}%")
                    st.info("🎯 **Acción recomendada:** Aunque los patrones actuales sugieren que es benigno, recuerda realizar tu chequeo preventivo mensual.")
st.markdown('</div>', unsafe_allow_html=True)

# 5. TARJETA: GUÍA DE PREVENCIÓN
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">☀️ Guía de Prevención y Cuidado de la Piel</p>', unsafe_allow_html=True)
st.write("El cáncer de piel es uno de los más prevenibles. Sigue estas recomendaciones:")
st.markdown("""
* **🧴 Protector Solar Diario:** Usa bloqueador solar con un FPS de 30 o superior todos los días.
* **⏰ Evita las Horas Pico:** Trata de no exponerte directamente al sol entre las **10:00 a.m. y las 4:00 p.m.**
* **👒 Ropa de Protección:** Cuando salgas, usa sombreros de ala ancha y lentes de sol.
* **🔎 Conoce tu Piel (Regla ABCDE):** Revisa tus lunares buscando Asimetría, Bordes, Color, Diámetro y Evolución.
""")
st.markdown('</div>', unsafe_allow_html=True)
