README.md
# 🩺 DermAI - Sistema de Cribado Preliminar de Lesiones Cutáneas

Este proyecto es una aplicación web interactiva diseñada en **Streamlit** que utiliza un modelo de Redes Neuronales Convencionales (CNN) entrenado en **TensorFlow** para analizar imágenes de lesiones en la piel y ofrecer una orientación preliminar.

## ⚠️ Aviso Importante (Disclaimer)
Esta herramienta es estrictamente de carácter **informativo y educativo**. No proporciona diagnósticos médicos definitivos ni sustituye la evaluación clínica de un dermatólogo profesional.

## 📁 Estructura del Proyecto
* `app.py`: Código principal de la interfaz de usuario.
* `dermai_modelo.h5`: Modelo de Inteligencia Artificial entrenado.
* `requirements.txt`: Lista de dependencias del sistema.

## 🚀 Cómo ejecutarlo localmente
1. Instalar las librerías necesarias:
   `pip install -r requirements.txt`
2. Iniciar la aplicación:
   `streamlit run app.py`