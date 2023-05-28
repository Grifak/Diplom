import random
import numpy as np
from skimage.transform import resize
from scipy.ndimage import rotate
from PIL import Image


def transform_image(image, size, name, i):
    # Изменение размера изображения
    if image.size == 0:
        print(f"FAIL {name} number {i}")

    print(f"Size = {image.size}")
    resized_image = resize(image, size, mode='reflect', anti_aliasing=True)

    # Отражение по горизонтали
    if random.choice([True, False]):
        resized_image = np.fliplr(resized_image)

    # Поворот изображения на случайный угол до 30 градусов
    angle = random.uniform(-30, 30)
    rotated_image = rotate(resized_image, angle, mode='reflect', reshape=False)

    # Изменение размера изображения до 256x256 пикселей
    final_image = resize(rotated_image, size, mode='reflect', anti_aliasing=True)

    return final_image


def data_generator(scans, input_shape):
    while True:
        # Инициализация пустых списков для хранения данных и меток
        X_batch = []
        y_batch = []

        # Итерация по выбранным индексам
        for scan in scans:
            # Получение случайного среза из скана
            i = random.randint(0, 319)
            slice = scan.get_slice(i)
            mask = scan.get_2D_mask()

            # Преобразование изображения
            transformed_slice = transform_image(slice, input_shape[:2], scan.filename+"slice", i)
            transformed_slice = np.expand_dims(transformed_slice, axis=0)  # Добавить размер пакета (batch size) в изображение
            transformed_slice = np.expand_dims(transformed_slice, axis=-1)

            transformed_mask = transform_image(mask, input_shape[:2], scan.filename+"mask", i)
            transformed_mask = np.expand_dims(transformed_mask, axis=0)  # Добавить размер пакета (batch size) в изображение
            transformed_mask = np.expand_dims(transformed_mask, axis=-1)


            # Добавление среза и маски в списки данных и меток
            X_batch.append(transformed_slice)
            y_batch.append(transformed_mask)

        # Преобразование списков в массивы
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        yield X_batch, y_batch


def get_training_data(SCANS):
    X_train = []  # Тренировочные изображения
    y_train = []  # Маски сегментации

    for OCT_data in SCANS:
        image = OCT_data.get_slice(200)  # Получение двухмерного изображения

        image = Image.fromarray(image)
        image = image.convert('L')

        # Преобразование размера изображения до 256x256
        image = image.resize((256, 256))

        mask = OCT_data.get_2D_mask()  # Получение маски сегментации

        # Преобразование размера маски до 256x256
        mask = mask.resize((256, 256))

        # Аугментация данных
        scale_factor = random.uniform(0.6, 1.4)  # Случайный масштабный коэффициент
        image = image.resize((int(256 * scale_factor), int(256 * scale_factor)))  # Изменение размера изображения
        mask = mask.resize((int(256 * scale_factor), int(256 * scale_factor)))  # Изменение размера маски

        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Отражение по горизонтали
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)  # Отражение маски по горизонтали

        angle = random.uniform(-30, 30)  # Случайный угол поворота
        image = image.rotate(angle)  # Поворот изображения
        mask = mask.rotate(angle)  # Поворот маски

        # Преобразование изображений в массивы NumPy
        image_array = np.array(image)
        mask_array = np.array(mask)

        # Нормализация значений пикселей изображений
        image_array = image_array / 255.0
        mask_array = mask_array / 255.0

        X_train.append(image_array)
        y_train.append(mask_array)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train