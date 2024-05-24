from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import data_handler

# Загрузка данных из CSV файла
data = pd.read_csv('data.csv')
data = data.sample(frac=1, random_state=42)

X, y = data_handler.handle_data(data)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Выставление параметров и обучение
model = MLPClassifier(hidden_layer_sizes=(75, 55), max_iter=300, activation='relu', solver='adam', random_state=42)
model.fit(X_train, y_train)

# Выгрузка обученной модели в файл
dump(model, 'model.joblib')

# Оценка производительности модели
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


