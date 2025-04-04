# 🧠 TensorFlow MNIST Digit Recognizer + Streamlit UI

Этот проект обучает модель для распознавания рукописных цифр (MNIST) с помощью TensorFlow и предоставляет веб-интерфейс на Streamlit.

## 🚀 Запуск проекта

### 1. Установите зависимости

```bash
pip install -r requirements.txt

2. Обучите модель
bash
Copy
Edit
python main.py
После этого появится файл mnist_model.h5


Откроется веб-интерфейс для загрузки изображения и предсказания цифры.

📂 Структура проекта
bash
Copy
Edit
.
├── app.py            # Веб-интерфейс
├── main.py           # Обучение модели
├── mnist_model.h5    # Сохраняется после обучения
├── requirements.txt
└── README.md

👨‍💻 Автор
Devlet, 2025

yaml
Copy
Edit

---

## ✅ Шаг 4. Добавь, закоммить и запушь

```bash
git add .
git commit -m "Initial commit: TensorFlow MNIST + Streamlit"
git push origin main

✅ Шаг 5. Запусти у себя
bash
Copy
Edit
# Установка зависимостей
pip install -r requirements.txt

# Обучение модели
python main.py

# Запуск UI
streamlit run app.py