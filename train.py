import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split



# Basic Parameters
size = (64, 64)
cat_img_list = []
dog_img_list = []
base_path = 'PetImages'

# 影像類別數
NUM_CLASSES = 2

# 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
BATCH_SIZE = 8

# 凍結網路層數
FREEZE_LAYERS = 2

# Epoch 數
NUM_EPOCHS = 5

# 模型輸出儲存的檔案
WEIGHTS_FINAL = 'model-resnet50-final.h5'

for root, dirs, files in os.walk(base_path):
    print('root = ' + str(root))
    print('dirs = ' + str(dirs))
    for file in files:
        if file.endswith('.jpg'):
            filename = os.path.join(root, file)
            file_size = os.path.getsize(filename)
            category_name = os.path.basename(root)
            if file_size >= 10240:
                im = Image.open(filename)
                if im.mode == 'RGB':
                    im = im.resize(size, Image.BILINEAR)
                    imarray = np.array(im)
                    imarray = (imarray - np.min(imarray))/(np.max(imarray) - np.min(imarray))
                    if category_name == 'Cat':
                        cat_img_list.append(imarray)
                    elif category_name == 'Dog':
                        dog_img_list.append(imarray)

cat_img_arr = np.array(np.float16(cat_img_list))
dog_img_arr = np.array(np.float16(dog_img_list))

cat_img_label = np.ones(cat_img_arr.shape[0])*0
dog_img_label = np.ones(dog_img_arr.shape[0])*1

img_arr = np.concatenate((cat_img_arr, dog_img_arr), axis=0)
img_label = np.concatenate((cat_img_label, dog_img_label), axis = 0)
img_label = keras.utils.to_categorical(img_label, num_classes=2)

temp = list(zip(img_arr, img_label))
random.shuffle(temp)
img_arr, img_label = zip(*temp)
img_arr = np.asarray(img_arr)
img_label = np.asarray(img_label)

print(img_arr)

train_data, test_data, train_label, test_label = train_test_split(img_arr, img_label, test_size=0.2, random_state=42)
train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2, random_state=42)


# 以訓練好的 ResNet50 為基礎來建立模型，
# 捨棄 ResNet50 頂層的 fully connected layers
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(size[0], size[1], 3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# 輸出整個網路結構
print(net_final.summary())
plot_model(net_final, to_file='model.png')

'''
# tensorboard
model_dir = 'lab2-logs/models'
os.makedirs(model_dir)
log_dir = os.path.join('lab2-logs', 'model-1')
model_cbk = TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(model_dir+'/Best-model-1.h5',
                                             monitor='val_mean_absolute_error',
                                             save_best_only=True,
                                             mode='min')
'''
# 訓練模型
net_final.fit(train_data, train_label, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

# 儲存訓練好的模型
net_final.save(WEIGHTS_FINAL)
