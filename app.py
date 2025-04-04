import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

def preprocess_image(image: Image.Image):
    image = image.convert("L").resize((28, 28))
    img_array = 255 - np.array(image)
    img_array = img_array / 255.0
    return img_array.reshape(1, 28, 28)

st.title("🧠 MNIST Распознавание Цифр")
uploaded_file = st.file_uploader("Загрузите изображение PNG/JPG", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Вход", use_column_width=False)

    model = load_model()
    processed_img = preprocess_image(image)

    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"Предсказание: {predicted_class}")
    st.caption(f"Уверенность: {confidence:.2f}")
