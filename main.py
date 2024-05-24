import cv2
import mediapipe as mp

import pandas as pd
from joblib import load
from vector import Vector
import data_handler


model = load('model.joblib') # Загрузка файла, содержащего обученную модель

cap = cv2.VideoCapture(0) # Переменная для захвата видео с камеры

# Настройка библиотеки Mediapipe для работы с распознаванием кистей
hands = mp.solutions.hands.Hands(static_image_mode=False,
                                 max_num_hands=1,
                                 min_tracking_confidence=0.5,
                                 min_detection_confidence=0.5)

mpDraw = mp.solutions.drawing_utils # Утилита Mediapipe для отрисовки соединений опорных точек кисти

# Функция отрисовки угаданного ответа
def write_answer(prediction, image):
    text = str(prediction) # Установка текста
    font = cv2.FONT_HERSHEY_SIMPLEX # Установка шрифта
    font_scale = 2 # Размер шрифта
    font_color = (255, 255, 255) # Цвет текста
    thickness = 3 # Толщина шрифта

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0] # Инициализация текста
    text_x = (image.shape[1] - text_size[0]) // 2 # X координата расположения текста
    text_y = (image.shape[0] - text_size[1]) # Y координата расположения текста
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness) # Отрисовка текста

def predict(v):
    data = pd.DataFrame() # Создание датафрейма
    # В цикле происходит полное заполнение датафрейма координатами точек
    for i in range(0, 20):
        data = pd.concat([data, pd.DataFrame({'x{}'.format(i + 1): v[i].x}, index=[0])],
                         axis=1)
        data = pd.concat([data, pd.DataFrame({'y{}'.format(i + 1): v[i].y}, index=[0])],
                         axis=1)
        data = pd.concat([data, pd.DataFrame({'z{}'.format(i + 1): v[i].z}, index=[0])],
                         axis=1)

    # Отметка типа жеста, как 'неизвестного'
    data = pd.concat([data, pd.DataFrame({'type': 'unknown'}, index=[0])],
                     axis=1)
    data, t_ = data_handler.handle_data(data) # Обработка датафрейма для использования в модели
    # print(data, model.predict(data))
    return model.predict(data) # Возврат ответа

while True:
    _, img = cap.read() # Захват изображения с камеры
    res = hands.process(img) # Обработка захваченного изображения в mediapipe
    if res.multi_hand_landmarks: # Проверка на наличие захваченной руки
        h, w, _ = img.shape # Взятие размеров захваченного изображения с камеры
        supportPoint = res.multi_hand_landmarks[0].landmark[0] # Взятие точки под индексом 0 (основание ладони)
        vectors = [] # Создание пустого списка для заполнения его векторами точек
        for id, lm in enumerate(res.multi_hand_landmarks[0].landmark): # Прохождение по списку точек, захваченной кисти
            # lm.x и lm.y положение точки кисти на экране от 0 до 1 в пределах камеры
            cx, cy = int(lm.x * w), int(lm.y * h) # Расчёт положения точки в координатном формате
            cv2.circle(img, (cx, cy), 20, (255, 0, 255)) # Отрисовка точки при помощи cv2
            # Добавление точки в массив относительно опорной точки в векторном формате
            vectors.append(Vector.landmark_to_vector(lm) - supportPoint)
        mpDraw.draw_landmarks(img, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS) # Отрисовка соединений кисти

        write_answer(predict(vectors), img) # Отображение угаданной буквы

    cv2.imshow("Hand Track", img) # Рендер всех наших отрисовок
    cv2.waitKey(1)