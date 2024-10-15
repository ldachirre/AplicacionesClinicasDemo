import streamlit as st

# Título de la aplicación en la barra lateral
st.sidebar.title("Navegación")

# Opciones de la barra lateral
menu = st.sidebar.radio("Selecciona una sección", 
                        ("Inicio", "Detección", "Información", "Acerca de la app"))

# Página de Inicio
if menu == "Inicio":
    st.title("Detección de Tumores en la Piel")
    st.write("""
    Bienvenido a la aplicación de detección de tumores en la piel.
    Esta herramienta utiliza inteligencia artificial para analizar imágenes de la piel y detectar
    posibles tumores.
    """)
    
    st.subheader("¿Cómo funciona?")
    st.write("""
    1. Sube una imagen de una parte de tu piel donde sospechas que hay una anomalía.
    2. La aplicación procesará la imagen utilizando un modelo de machine learning.
    3. Obtendrás un resultado indicando si es necesario que consultes a un dermatólogo.
    """)
    
    st.subheader("Ejemplo de uso")
    st.write("Aquí se mostraría un ejemplo de cómo utilizar la aplicación.")
    st.image("https://via.placeholder.com/300", caption="Ejemplo de imagen de piel")

# Página de Detección
elif menu == "Detección":
    st.title("Detección de Tumores en la Piel")
    
    st.subheader("Sube una imagen")
    image_file = st.file_uploader("Selecciona una imagen de tu piel", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        st.image(image_file, caption="Imagen cargada", use_column_width=True)
        st.write("Procesando la imagen...")
        
        # Simulación del análisis de la imagen
        import time
        time.sleep(2)
        st.success("Predicción: Melanoma sospechoso. Por favor, consulta a un dermatólogo.")
    
    st.warning("Nota: Esta aplicación no reemplaza una consulta médica profesional.")

# Página de Información
elif menu == "Información":
    st.title("Información sobre el Cáncer de Piel")
    
    st.subheader("Tipos de cáncer de piel")
    st.write("""
    - **Melanoma**: Tipo de cáncer de piel más peligroso. Puede propagarse a otras partes del cuerpo.
    - **Carcinoma de células basales**: Crecimiento lento, generalmente no se propaga.
    - **Carcinoma de células escamosas**: Puede ser más agresivo y propagarse si no se trata.
    """)
    
    st.subheader("Recomendaciones")
    st.write("""
    - Usa protector solar diariamente.
    - Realiza autoexámenes regulares de la piel.
    - Consulta a un dermatólogo si notas algún cambio en lunares o manchas.
    """)

# Página de Acerca de la App
elif menu == "Acerca de la app":
    st.title("Acerca de la Aplicación")
    
    st.write("""
    Esta aplicación fue desarrollada con el propósito de ayudar a las personas a detectar posibles tumores en la piel
    de forma temprana. Utiliza un modelo de inteligencia artificial entrenado en un conjunto de datos de imágenes médicas.
    
    **Tecnologías utilizadas**:
    - Python
    - Streamlit
    - Modelos de Machine Learning para la detección de cáncer de piel
    
    **Créditos**:
    - Desarrollador: Luis Danilo Chirre Arias
    - Dataset: HAM10000
    """)
    
    st.subheader("Contacto")
    st.write("Para más información, puedes contactarnos en: [email@example.com](mailto:email@example.com)")

