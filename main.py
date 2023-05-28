from read_dataset import *
from analize_abnorm import *
from UNet_TensorFlow import *
from data_processor import *


anno_fld = '/Users/notremembering/Desktop/диплом/annotations_01_06_2019'
dumps = [os.path.join(anno_fld, i) for i in os.listdir(anno_fld) if i.endswith('.dump')]
not_centered = ['CEA OD', 'GKA OS', 'GMS OS', 'HVA OS', 'MVE OD', 'SYA OS']

background_color = '0,0,0'
label_color = ['grade1:255,0,0', 'grade2:0,255,0', 'grade3:0,0,255']
mask_bitness = 24 # (8, 24), 8 for binary

SCANS = read_dataset(anno_fld)

# Предварительно обработанные данные
# Инициализация пустых списков для хранения данных и меток
batch_size = 2
input_shape = (256, 256, 1)
epochs = 10

model = unet_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_size = int(0.8 * len(SCANS))
train_scans = SCANS[:train_size]
val_scans = SCANS[train_size:]

# Создание генераторов данных для обучения и проверки
train_data_generator = data_generator(train_scans, input_shape)
val_data_generator = data_generator(val_scans, input_shape)

# Обучение модели
model.fit(
    train_data_generator,
    steps_per_epoch=len(train_scans) // batch_size,
    validation_data=val_data_generator,
    validation_steps=len(val_scans) // batch_size,
    # validation_split=0.2,
    epochs=epochs
)