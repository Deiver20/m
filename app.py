import gradio as gr
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np

# Asegurarse de tener los recursos de nltk descargados
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Cargar el modelo previamente entrenado
model = load_model("modelo_sentimientos.h5")

# Parámetros de preprocesamiento
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Función para limpiar el texto
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Eliminar HTML
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Eliminar caracteres especiales
    text = text.lower()  # Convertir a minúsculas
    tokens = word_tokenize(text)  # Tokenización
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatización y eliminación de stopwords
    return ' '.join(tokens)

# Función para hacer predicciones
def predict_sentiment(review):
    cleaned_review = clean_text(review)
    # Aquí necesitas convertir cleaned_review en la forma que tu modelo espera.
    # Esto normalmente requiere un tokenizador.
    # En esta versión, solo estamos imprimiendo la reseña limpia.
    print("Reseña limpia:", cleaned_review)

    # Convertir cleaned_review a una secuencia numérica o a una representación que el modelo pueda usar.
    # Esto es solo un ejemplo, ya que sin un tokenizador, no puedes hacer la predicción.
    # prediction = model.predict(...)  # Aquí deberías incluir la lógica de predicción.

    sentiment = 'Positivo'  # Cambia esto por el resultado de la predicción real.
    return sentiment

# Crear la interfaz con Gradio con diseño mejorado
with gr.Blocks() as interface:
    gr.Markdown(
        """
        # 🧠 Clasificación de Sentimientos
        Esta aplicación predice si una reseña es **Positiva** o **Negativa**. Simplemente ingresa tu texto en el cuadro de abajo.
        """
    )

    # Input de reseña
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Escribe tu reseña:")
            review_input = gr.Textbox(lines=3, placeholder="Escribe una reseña...", label="Reseña de producto")

    # Output de resultado
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Resultado:")
            result_output = gr.Textbox(label="Sentimiento")

    # Botón para hacer la predicción
    with gr.Row():
        submit_button = gr.Button("Predecir Sentimiento")

    submit_button.click(fn=predict_sentiment, inputs=review_input, outputs=result_output)

    # Información adicional
    gr.Markdown(
        """
        ---
        **Clasificación de Sentimientos** usa procesamiento de lenguaje natural para analizar el texto. ¡Prueba diferentes reseñas y observa los resultados!
        """
    )

# Iniciar la interfaz
interface.launch()
