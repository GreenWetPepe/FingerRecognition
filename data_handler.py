import pandas as pd

# Файл для одинаковой обработки датасета на стадиях предугадывания и обучения модели

# Точки кисти, использующиеся в обучении
delf_point_id = [4, 8, 12, 16, 20]
start_point_id = [2, 5, 8, 13, 17]

# Функция преобразования датафрейма в необходимый формат
def handle_data(data):
    d = pd.DataFrame()
    d_ans = data['type'] # Отделение разметки
    for i in range(0, len(delf_point_id)): # Цикл взятия необходимых нам точек
        d['x{}'.format(start_point_id[i])] = data['x{}'.format(start_point_id[i])]
        d['y{}'.format(start_point_id[i])] = data['y{}'.format(start_point_id[i])]
        d['x{}'.format(delf_point_id[i])] = data['x{}'.format(delf_point_id[i])]
        d['y{}'.format(delf_point_id[i])] = data['y{}'.format(delf_point_id[i])]

        # Нахождение разности между точкой начала пальца и его конца, то есть дельта по осям X и Y
        d['diff_x{}'.format(delf_point_id[i])] = data['x{}'.format(delf_point_id[i])]\
                                                 - data['x{}'.format(start_point_id[i])]
        d['diff_y{}'.format(delf_point_id[i])] = data['y{}'.format(delf_point_id[i])]\
                                                 - data['y{}'.format(start_point_id[i])]
    return d, d_ans


