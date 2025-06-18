import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# ==================== 1. Создаём искусственные данные ====================
random_data_size = 100

data = {
    "user_id": range(1, random_data_size + 1),
    "topic_completed": np.random.choice(["variables", "functions", "lists", "OOP", "decorators"], random_data_size),
    "errors": np.random.randint(1, 10, random_data_size),
    "time_spent": np.random.randint(10, 60, random_data_size),
    "difficulty": np.random.choice(["easy", "medium", "hard"], random_data_size),
    "next_topic": np.random.choice(["functions", "lists", "OOP", "decorators", "generators"], random_data_size)
}

df = pd.DataFrame(data)


# ==================== 2. Предобработка данных ====================
# Кодируем текст в числа
label_encoders = {}
for column in ["topic_completed", "difficulty", "next_topic"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Разделяем данные
X = df[["topic_completed", "errors", "time_spent", "difficulty"]]
y = df["next_topic"]

# Нормализуем
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==================== 3. Создаём и обучаем модель ====================
model = Sequential([
    Dense(128, activation='selu', input_shape=(X_train.shape[1],)),
    #Dropout(0.3),
    Dense(64, activation='selu'),
    Dense(32, activation='selu'),
    Dense(len(label_encoders["next_topic"].classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=300,
                    batch_size=30,
                    validation_data=(X_test, y_test),
                    verbose=1)


# ==================== 4. Функция для рекомендаций ====================
def recommend_next_topic(topic_completed, errors, time_spent, difficulty):
    # Кодируем входные данные
    topic_encoded = label_encoders["topic_completed"].transform([topic_completed])[0]
    difficulty_encoded = label_encoders["difficulty"].transform([difficulty])[0]

    # Нормализуем
    input_data = scaler.transform([[topic_encoded, errors, time_spent, difficulty_encoded]])

    # Предсказываем
    probabilities = model.predict(input_data, verbose=0)[0]
    predicted_topic_idx = probabilities.argmax()

    # Декодируем название темы
    return label_encoders["next_topic"].inverse_transform([predicted_topic_idx])[0]


# ==================== 5. Пример использования ====================
# Выведем оригинальные данные для наглядности
print("Первые 5 строк исходных данных:")
print(df.head())

# Тестируем рекомендацию
print("\nРекомендация для ученика, который изучил 'functions', сделал 4 ошибки, потратил 30 минут, сложность 'medium':")
print(recommend_next_topic("functions", 4, 30, "medium"))

# ==================== 6. Визуализация точности ====================
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()