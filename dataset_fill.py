import keyboard
import csv
import cv2
import mediapipe as mp
from vector import Vector

cap = cv2.VideoCapture(0) # Переменная для захвата видео с камеры

# Настройка библиотеки Mediapipe для работы с распознаванием кистей
hands = mp.solutions.hands.Hands(static_image_mode=False,
                                 max_num_hands=1,
                                 min_tracking_confidence=0.5,
                                 min_detection_confidence=0.5)

mpDraw = mp.solutions.drawing_utils # Утилита Mediapipe для отрисовки соединений опорных точек кисти

# Функция преобразования списка векторов в список координат точек
def data_to_list(v):
    line = []
    if res.multi_hand_landmarks:
        for i in range(1, 20):
            line.append(v[i].x)
            line.append(v[i].y)
            line.append(v[i].z)
    return line

# Функция записи жеста в csv файл
def hands_to_csv_data(line):
    line.append("A") # Заранее указываем разметку жеста, который хотим записать
    print(line)
    with open("data.csv", 'a', newline='') as file: # Открываем файл для записи
        writer = csv.writer(file)
        writer.writerow(line) # Производим запись строки

def on_key_event(event):
    if event.event_type == keyboard.KEY_UP and event.name == 'f':  # Проверяем, что это событие нажатия клавиши
        # Проводим запись точек
        vectors = []
        supportPoint = res.multi_hand_landmarks[0].landmark[0]
        for id, lm in enumerate(res.multi_hand_landmarks[0].landmark):
            vectors.append(Vector.landmark_to_vector(lm) - supportPoint)
        line = data_to_list(vectors) # Перемещаем вектор в список
        if len(line) != 0: # Если длина ненулевая, то записываем в csv файл
            hands_to_csv_data(line)

keyboard.on_release(on_key_event) # Добавление события, если кнопку отпустили


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

    cv2.imshow("Hand Track", img) # Рендер всех наших отрисовок
    cv2.waitKey(1)