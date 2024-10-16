import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import FixCapsNet  # Asegúrate que el archivo model.py esté en el mismo directorio

# Definir una función para cargar el modelo
@st.cache(allow_output_mutation=True)
def load_model():
    n_channels = 3  # Asumiendo que las imágenes son RGB
    conv_outputs = 128  # Ajusta esto según tu modelo
    num_primary_units = 8  # Ajusta esto según tu modelo
    primary_unit_size = 16 * 6 * 6  # Esto también debe coincidir con tu modelo
    n_classes = 7  # Número de clases en tu caso específico
    mode = 'DS'  # Modo según tu modelo
    
    # Inicializar la arquitectura del modelo
    model = FixCapsNet(conv_inputs=n_channels, 
                       conv_outputs=conv_outputs,
                       primary_units=num_primary_units,
                       primary_unit_size=primary_unit_size,
                       num_classes=n_classes,
                       output_unit_size=16,
                       init_weights=True,
                       mode=mode)
    
    # Cargar los pesos entrenados
    state_dict = torch.load('modelo.pth', map_location=torch.device('cpu'))

    # Filtrar las claves inesperadas
    filtered_state_dict = {k: v for k, v in state_dict.items() if not any(x in k for x in ['total_ops', 'total_params'])}
    
    # Cargar el state_dict filtrado en el modelo
    model.load_state_dict(filtered_state_dict, strict=False)  # strict=False para evitar problemas con claves faltantes
    
    model.eval()  # Cambiar a modo evaluación
    return model

# Cargar el modelo
model = load_model()

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
        # Convertir la imagen cargada a formato PIL
        image = Image.open(image_file)
        
        # Mostrar la imagen cargada
        st.image(image, caption="Imagen cargada", use_column_width=True)
        
        # Transformar la imagen para que sea compatible con el modelo
        transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ajusta el tamaño según la arquitectura del modelo
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalización típica de imágenes preentrenadas
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Añadir dimensión para el batch
    
    # Realizar la predicción
    with torch.no_grad():
        output = model(image_tensor)
        v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True))
        pred = v_mag.data.max(1, keepdim=True)[1].cpu()  # Obtener la clase predicha
        
    # Mostrar el resultado de la predicción
    st.write(f"Lesión predicha: {pred.item()}")  # Muestra la clase predicha    
    
    #st.success("Predicción: Melanoma sospechoso. Por favor, consulta a un dermatólogo.")   
    
    
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

