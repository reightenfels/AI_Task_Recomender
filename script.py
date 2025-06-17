import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
import pandas as pd
from typing import List, Dict, Tuple


class TaskRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        self.tasks = pd.DataFrame(columns=['id', 'text', 'difficulty', 'solved'])
        self.task_vectors = None

    def add_tasks(self, tasks: List[Dict]):
        """Добавляет новые задачи в систему."""
        new_data = pd.DataFrame(tasks)
        self.tasks = pd.concat([self.tasks, new_data], ignore_index=True)

        if len(self.tasks) > 0:
            self.task_vectors = self.vectorizer.fit_transform(self.tasks['text'])

    def train_difficulty_model(self, X: List[str], y: List[int]):
        """Обучает модель оценки сложности."""
        X_vec = self.vectorizer.transform(X)
        self.mlp.fit(X_vec, y)

    def get_recommendations(self, solved_task_ids: List[int], n_recommendations: int = 3) -> List[Tuple[int, float]]:
        """Рекомендует задачи на основе решённых."""
        if len(self.tasks) == 0 or self.task_vectors is None:
            return []

        # Получаем индексы решённых и нерешённых задач
        solved_indices = self.tasks[self.tasks['id'].isin(solved_task_ids)].index
        unsolved_indices = self.tasks[~self.tasks['id'].isin(solved_task_ids)].index

        if len(solved_indices) == 0:
            # Если нет решённых, рекомендуем случайные
            unsolved_tasks = self.tasks.iloc[unsolved_indices].sample(min(n_recommendations, len(unsolved_indices)))
            return list(zip(unsolved_tasks['id'], [1.0] * len(unsolved_tasks)))

        # Вычисляем средний вектор решённых задач и преобразуем в массив
        solved_vectors = self.task_vectors[solved_indices]
        avg_solved_vector = np.asarray(np.mean(solved_vectors, axis=0))  # Ключевое исправление

        # Вычисляем сходство с нерешёнными задачами
        unsolved_vectors = self.task_vectors[unsolved_indices]
        similarities = cosine_similarity(avg_solved_vector.reshape(1, -1), unsolved_vectors).flatten()

        # Выбираем топ-N рекомендаций
        unsolved_ids = self.tasks.iloc[unsolved_indices]['id'].values
        top_indices = np.argsort(similarities)[-n_recommendations:][::-1]

        return [(unsolved_ids[i], similarities[i]) for i in top_indices]


# Пример использования
if __name__ == "__main__":
    recommender = TaskRecommender()

    tasks = [
        {"id": 1, "text": "Write a function to add two numbers.", "difficulty": 1, "solved": False},
        {"id": 2, "text": "Implement a stack using lists in Python.", "difficulty": 2, "solved": False},
        {"id": 3, "text": "Create a recursive Fibonacci function.", "difficulty": 3, "solved": False},
        {"id": 4, "text": "Write a Python decorator to measure function execution time.", "difficulty": 4,
         "solved": False},
    ]
    recommender.add_tasks(tasks)

    solved_ids = [1, 2]
    recommendations = recommender.get_recommendations(solved_ids)

    print("Рекомендуемые задачи:")
    for task_id, similarity in recommendations:
        task = next(t for t in tasks if t["id"] == task_id)
        print(f"ID: {task_id}, Сходство: {similarity:.2f}, Текст: {task['text']}")