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
from tensorflow.keras.preprocessing.image import array_to_img
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

train_data, test_data, train_label, test_label = train_test_split(img_arr, img_label, test_size=0.2, random_state=42)
train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2, random_state=42)


model = load_model(WEIGHTS_FINAL)

label_name = {0:'Cat', 1:'Dog'}
index = np.random.randint(1, 200, size=1, dtype=np.int32)
image = test_data[index]
image_show = array_to_img(image)
img = image.reshape(-1, 64, 64, 3)
image = img.reshape(64, 64, 3)
image = np.int32(image*256)

print(img)
pred = model.predict(img)
print('This photo is a {}, predicted as a {}'.format(label_name[np.argmax(test_label[index])], label_name[np.argmax(pred)]))
plt.imshow(image)
plt.title('This photo is a {}, predicted as a {}'.format(label_name[np.argmax(test_label[index])], label_name[np.argmax(pred)]))
plt.xticks([])
plt.yticks([])
plt.show()
