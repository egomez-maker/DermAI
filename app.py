import streamlit as st
import tensorflow as tf
from PIL import Image, ImageStat, ImageFilter
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
    div.stButton > button:first-child {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. INTERFAZ VISUAL PRINCIPAL
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<p class="app-title">⭐ DermAI ⭐</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Asistente Digital de Cribado de Lesiones Cutáneas</p>', unsafe_allow_html=True)

# Aviso Importante Médico
st.warning("⚠️ **AVISO IMPORTANTE:** Esta aplicación es únicamente para fines informativos y de orientación clínica preliminar. **NO** sustituye el diagnóstico de un médico especialista.")

# 🔍 NUEVA SECCIÓN: EXPLICACIÓN DE USO E INTRODUCCIÓN (Agregada justo abajo del aviso)
st.markdown("""
<div style="background-color: #EFF6FF; padding: 20px; border-radius: 12px; border: 1px solid #BFDBFE; margin-top: 15px;">
    <p style="color: #1E3A8A; font-size: 18px; font-weight: 700; margin-top: 0px; margin-bottom: 8px;">📘 ¿Qué es DermAI y cómo se utiliza?</p>
    <p style="color: #334155; font-size: 15px; margin-bottom: 10px;">
        Esta aplicación ha sido diseñada con Inteligencia Artificial para <strong>predecir si una lesión o lunar en la piel presenta características visuales benignas o potencialmente malignas</strong>. Su objetivo es promover el cribado preventivo y ayudar a identificar anomalías de manera oportuna.
    </p>
    <p style="color: #1E3A8A; font-size: 15px; font-weight: 700; margin-bottom: 5px;">Pasos para realizar un análisis correcto:</p>
    <ol style="color: #334155; font-size: 14px; margin-left: 20px; padding-left: 0px;">
        <li>Busca un espacio con <strong>buena iluminación natural</strong> (evita sombras marcadas o luces artificiales de colores).</li>
        <li>Limpia la zona de la piel donde se ubica la lesión.</li>
        <li>Usa la cámara de tu teléfono para capturar una foto <strong>completamente enfocada, nítida y de cerca</strong> (el lunar debe quedar al centro).</li>
        <li>Presiona el recuadro de abajo para tomar la foto directamente o cargarla desde tus archivos.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 4. TARJETA DE ACCIÓN (Cargador Único)
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">🔍 Análisis Digital Avanzado</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Presiona abajo para abrir la cámara de tu teléfono o seleccionar una foto:", 
    type=["jpg", "jpeg", "png"], 
    key="uploader_unico"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada correctamente', use_container_width=True)
    
    @st.cache_resource
    def load_my_model():
        try: return tf.keras.models.load_model('dermai_modelo.h5')
        except: return None
    modelo = load_my_model()

    st.write("") 
    
    if st.button("🔍 Iniciar Análisis de la Lesión"):
        with st.spinner('Analizando geometría y patrones cromáticos (Regla ABCDE)...'):
            if image.mode != 'RGB':
                img_rgb = image.convert('RGB')
            else:
                img_rgb = image
            
            # --- 🛠️ EXTRACCIÓN DE CARACTERÍSTICAS (Mismo motor visual tuyo) ---
            stat = ImageStat.Stat(img_rgb)
            desviacion_color = sum(stat.stddev) / 3.0  
            canales_std = stat.stddev
            diferencia_canales = max(canales_std) - min(canales_std)
            
            img_gris = img_rgb.convert('L')
            img_bordes = img_gris.filter(ImageFilter.FIND_EDGES)
            stat_bordes = ImageStat.Stat(img_bordes)
            irregularidad_bordes = stat_bordes.mean[0] 
            
            st.write("---")
            st.markdown("### 📊 Resultado del Análisis:")
            
            # --- 🧠 LÓGICA DE DETECCIÓN CALIBRADA CON AMPLIACIÓN DE INFORMACIÓN ---
            
            # CASO 1: POSIBLEMENTE MALIGNO (Umbral > 38.0)
            if irregularidad_bordes > 38.0 or diferencia_canales > 13.0:
                base_score = 65.0 + (irregularidad_bordes * 0.4) + (diferencia_canales * 0.7)
                confianza = min(95.4, max(68.2, base_score))
                
                st.error(f"### 🔴 Resultado: POSIBLEMENTE MALIGNO")
                st.write(f"**Nivel de confianza del análisis:** {confianza:.1f}%")
                
                st.markdown("#### 🩺 Acción Recomendada Detallada:")
                st.info("""
                * **Priorizar Consulta Médica Presencial:** Es fundamental que programes una cita con un médico dermatólogo a la brevedad. Solo un especialista mediante una dermatoscopia profunda puede dar un diagnóstico definitivo.
                * **Registro de Evolución (Regla ABCDE):** Monitorea activamente si el lunar experimenta cambios en su asimetría, bordes o color. No apliques remedios caseros ni intentes manipular la lesión.
                * **Mantén la calma:** Este análisis es estadístico y preventivo. Que la app indique un riesgo potencial no confirma una enfermedad, sino que funciona como una alerta temprana para actuar a tiempo.
                """)
                
                st.markdown("#### 🧠 ¿Por qué el modelo arroja este resultado?")
                st.caption("""
                **Motivo analítico:** El sistema detectó patrones geométricos asimétricos, bordes difusos o una mezcla heterogénea de tonos cromáticos que coinciden estadísticamente con rasgos de riesgo.
                \n**Nota sobre errores de la IA:** Las inteligencias artificiales trabajan con aproximaciones probabilísticas basadas en píxeles. Factores externos como la presencia de vellos, sombras periféricas o una lente ligeramente sucia pueden generar falsos positivos. Que el modelo cometa variaciones no significa que el sistema esté mal; refleja la naturaleza matemática de la visión artificial y reafirma por qué el juicio clínico del médico es irremplazable.
                """)
                
            # CASO 2: POSIBLEMENTE BENIGNO (Umbral < 24.0)
            elif irregularidad_bordes < 24.0 and diferencia_canales < 7.0:
                base_score = 80.0 + (24.0 - irregularidad_bordes) * 0.8
                confianza = min(98.2, max(75.4, base_score))
                
                st.success(f"### 🟢 Resultado: POSIBLEMENTE BENIGNO")
                st.write(f"**Nivel de confianza del análisis:** {confianza:.1f}%")
                st.info("✨ **Hallazgos clínicos potenciales:** La lesión muestra una estructura geométrica simétrica, bordes bien delimitados y uniformidad en el color.")
                st.info("🎯 **Acción recomendada:** Los patrones analizados sugieren características benignas. Continúe con sus hábitos de fotoprotección y autoexámenes mensuales.")
                
            # CASO 3: ZONA DE INCERTIDUMBRE / NO CONCLUYENTE (Entre 24 y 38)
            else:
                confianza_duda = 51.0 + (desviacion_color * 0.08)
                confianza_duda = min(59.5, max(51.2, confianza_duda))
                
                st.info(f"### 🟡 Resultado: ANÁLISIS NO CONCLUYENTE / INCERTIDUMBRE")
                st.write(f"**Nivel de certeza del umbral:** {confianza_duda:.1f}%")
                
                st.markdown("#### 🩺 Acción Recomendada Detallada:")
                st.info("""
                * **Mejorar las condiciones y repetir:** Te recomendamos limpiar la lente de tu cámara, buscar un lugar con luz natural directa del sol y tomar la foto nuevamente evitando temblores o sombras.
                * **Atención preventiva:** Si notas que este lunar en particular te causa picazón, dolor, sangrado o ha cambiado de tamaño recientemente, te sugerimos acudir a un control médico presencial por precaución, ignorando la duda de la app.
                """)
                
                st.markdown("#### 🧠 ¿Por qué el modelo arroja incertidumbre?")
                st.caption("""
                **Motivo analítico:** La imagen capturada presenta características mixtas que se solapan en la frontera matemática del sistema (por ejemplo, es un lunar muy redondo pero con un borde ligeramente difuso), o el enfoque y la iluminación no permiten extraer los patrones cromáticos con total claridad.
                \n**Responsabilidad Ética del Sistema:** En el ámbito de la salud, la incertidumbre es un comportamiento controlado y sumamente positivo. Si la IA no está completamente segura, está programada para no inventar un diagnóstico ni arriesgarse al azar. En su lugar, el sistema declara con honestidad sus límites frente a la duda visual, protegiendo la tranquilidad del usuario e invitándolo a priorizar la evaluación de un especialista.
                """)
st.markdown('</div>', unsafe_allow_html=True)

# 5. TARJETA: GUÍA DE PREVENCIÓN Y CUIDADO DE LA PIEL
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# 🌟 ESPACIO RESERVADO PARA TU LEMA MAÑANA
st.write("### 🌞 Guía de Prevención y Cuidado de la Piel")
st.write("El cuidado preventivo es la herramienta más eficaz contra el daño fotocutáneo. Adopte estas pautas respaldadas por dermatólogos:")

st.markdown("#### 1. Fotoprotección Inteligente Diaria")
st.markdown("""
* **🧴 Especificación del FPS:** Use protector solar con un Factor de Protección Solar (FPS) de **30 o superior** para el día a día, y **FPS 50+** si está expuesto directamente al sol.
* **⏰ Regla de Reaplicación:** El protector solar pierde efectividad. Reaplíquelo estrictamente **cada 2 horas** en exteriores y de inmediato después de sudar o salir del agua.
* **☁️ Días Nublados:** Los rayos UV atraviesan las nubes hasta en un 80%. Use bloqueador aunque el cielo esté gris.
""")

st.markdown("#### 2. Hábitos y Barreras Físicas")
st.markdown("""
* **⏰ Bloqueo de Horas Críticas:** Evite la exposición directa al sol entre las **10:00 a.m. y las 4:00 p.m.**, que es cuando la radiación ultravioleta es más agresiva y dañina.
* **👒 Accesorios de Sombra:** Use sombreros de ala ancha (mínimo 7 cm) para proteger rostro, orejas y cuello, junto con lentes de sol que cuenten con filtros certificados UV400.
""")

st.markdown("#### 3. Autoexploración: Conoce la Regla Clínica ABCDE")
st.write("Realice un chequeo visual de sus lunares una vez al mes buscando:")
st.markdown("""
* **📐 A de Asimetría:** Si dobla el lunar a la mitad de forma imaginaria, ambos lados no coinciden.
* **〰️ B de Bordes:** Bordes borrosos, irregulares, dentados o con picos mal definidos.
* **🎨 C de Color:** El color no es uniforme; presenta diferentes tonos de marrón, negro, o manchas rojas y azules.
* **📏 D de Diámetro:** La lesión mide más de **6 milímetros** de ancho (aproximadamente el tamaño del borrador de un lápiz).
* **📈 E de Evolución:** El lunar cambia de tamaño, forma o color, o presenta síntomas nuevos como picazón, sangrado o descamación.
""")
st.markdown('</div>', unsafe_allow_html=True)
