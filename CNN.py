import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv3D, AveragePooling3D, Flatten
from data_loader_2_prova import DataGenerator
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import sys
import math as m

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

csv_logger = CSVLogger('log_200ep.csv', append=True, separator=';')

plt.ioff()

outputs = np.loadtxt(
    "/home/lbrodoloni/Larger_Dataset/difference_gs_summed_energies.txt")
# Standardizzo gli output in modo di abbassare il MAE
outputs = (outputs - outputs.mean()) / outputs.std()
# Normalizzo gli outputs
outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
no_data = int(len(outputs))

# Array contenenti gli indici delle istanze di train e di validation
no_data = int(len(outputs))
no_train_data = m.ceil(no_data * 0.9)
no_val_data = int(no_data - no_train_data)
train = [str(i) for i in range(no_train_data)]
validation = [str(i) for i in range(no_train_data, no_data)]
labels = {}
for i in range(no_data):
    labels[str(i)] = outputs[i]

partition = {'train': train, 'validation': validation}

params = {'dim': (50, 50, 50),
          'ydim': 1,
          'batch_size': 32,
          'n_channels': 1,
          'shuffle': True}

params_val = {'dim': (50, 50, 50),
          'ydim': 1,
          'batch_size': 1,
          'n_channels': 1,
          'shuffle': False}

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params_val)


cnn = Sequential()
cnn.add(
    Conv3D(8, 4, strides=2, padding='valid', activation='elu',
           data_format='channels_last', input_shape=(50, 50, 50, 1)))

cnn.add(Conv3D(10, 3, strides=1, padding='valid', activation='elu'))
cnn.add(AveragePooling3D(2))

#cnn.add(Conv3D(30, 3, strides=1, padding='valid', activation='elu'))
#cnn.add(AveragePooling3D(2))

cnn.add(Conv3D(30, 2, strides=1, padding='valid', activation='elu'))
cnn.add(AveragePooling3D(2))

cnn.add(Flatten())

cnn.add(Dense(units=20, activation='elu'))
cnn.add(Dense(units=10, activation='elu'))
cnn.add(Dense(units=5, activation='elu'))
#cnn.add(Dense(units=5, activation='elu'))
cnn.add(Dense(units=1))

cnn.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mae'])
cnn.summary()

cnn.fit(x=training_generator, validation_data=validation_generator, epochs=200, callbacks=[csv_logger], workers=4)

score = cnn.evaluate(validation_generator)
print(f'Validation Loss: {score[0]}\n')
print(f'\nValidation MAE: {score[1]}')
print(f'score_size: {sys.getsizeof(score)}')
print(f'cnn size: {sys.getsizeof(cnn)}')
print(f'cnn.fit size: {sys.getsizeof(cnn.fit)}')
print(f'data generator size: {sys.getsizeof(DataGenerator)}')
#  Returns an array which have a shape of (no_data, 1) containing prediction on val_generator
val_predictions = cnn.predict(validation_generator)
val_outputs = outputs[no_train_data:]

print(f'First ten theoretical outputs:\n')
for i in range(10):
	print(val_outputs[i])

print(f'\nFirst ten predictions:\n')
for i in range(10):
	print(val_predictions[i]) 
R2_val = r2_score(val_outputs, val_predictions)
print(f'R2_val = {R2_val}')
