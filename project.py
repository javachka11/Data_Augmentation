from google.colab import drive
drive.mount("/content/drive/")

from keras import layers
from keras import models
import numpy as np
import random

model = models.Sequential()

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same', input_shape=(250, 250, 3)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', input_dim = 7*7*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-6),
              metrics=['acc'])
              
def func(x):
    ins1 = random.uniform(-200, 200)
    ins2 = random.uniform(-200, 200)
    ins3 = random.uniform(-200, 200)
    z = x.copy()
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        for k in range(x.shape[2]):
          if k == 0:
            z[i, j, k] += ins1
          elif k == 1:
            z[i, j, k] += ins2
          elif k == 2:
            z[i, j, k] -= ins3
    return z

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

train_dir = '/content/drive/MyDrive/CleanedDirty/train'
validation_dir = '/content/drive/MyDrive/CleanedDirty/validation'
gen1_dir = '/content/drive/MyDrive/CleanedDirty/train/cleaned'
gen2_dir = '/content/drive/MyDrive/CleanedDirty/train/dirty'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.25, 
                                   height_shift_range=0.25,
                                   brightness_range=[0.35,0.6],
                                   zoom_range=[0.7, 1.3],
                                   channel_shift_range=150.0,
                                   preprocessing_function = func,
                                   fill_mode = 'nearest')


for i in range(20):
  if i < 10:
    pic = load_img('/content/drive/MyDrive/CleanedDirty/temp_train/cleaned/000' + str(i) + '.jpg')
  else:
    pic = load_img('/content/drive/MyDrive/CleanedDirty/temp_train/cleaned/00' + str(i) + '.jpg')
  pic_array = img_to_array(pic)
  pic_array = pic_array.reshape((1,) + pic_array.shape)
  count = 0
  for batch in train_datagen.flow(pic_array, batch_size = 1, save_to_dir = gen1_dir, save_prefix="cleaned."):
      count += 1
      if count == 100:
          break
          
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(250, 250),
                                                    batch_size=50,
                                                    class_mode='binary',
                                                    shuffle=True)


validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(250, 250),
                                                        batch_size=5,
                                                        class_mode='binary')

history = model.fit(train_generator,
                    steps_per_epoch=40,
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=4)
                    
model.save('/content/drive/MyDrive/project10.h5')            
