import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras import backend as K 
from keras.layers import Dense, Conv3D, AveragePooling3D, Flatten, MaxPooling3D
from dataloader import DataGenerator
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import sys
import math as m
import time
from datetime import datetime
import random
from iteration_utilities import duplicates

start_time = datetime.now()

SEED = 42
BATCH = 16
EPOCHS = 200

# Reproducibility 
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

plt.ioff()

outputs = np.loadtxt(
    "/home/lbrodoloni/Larger_Dataset/32grid_pot/all_dataset_train_32/difference_gs_summed_energies.txt")

no_data = int(len(outputs))
temp = np.arange(no_data) 
np.random.shuffle(temp) # Lista di indici shufflati 

no_train_data = m.ceil(no_data * 0.9) # varia la percentuale per variare la taglia del dataset 

no_val_data = m.ceil(no_train_data * 0.1) # il 10% del numero di dati di train li uso come numero di dati di test 

NAME = f"TEST-cnn_adattata-{no_train_data}train-{no_val_data}val-{int(time.time())}"

train = [str(i) for i in temp[:no_train_data]]
validation = [str(i) for i in temp[no_train_data:no_train_data + no_val_data]]

labels = {}
for i in temp[:no_train_data + no_val_data]:
    labels[str(i)] = outputs[i]
partition = {'train': train, 'validation': validation}
# Controllo che non ci siano duplicati tra i dati di train e quelli di test
prova = train + validation
print(len(list(duplicates(prova))))

outputs_val = []
for i in temp[no_train_data:no_train_data + no_val_data]:
    outputs_val.append(outputs[i])
outputs_val = np.array(outputs_val)

params = {'dim': (32, 32, 32),
          'ydim': 1,
          'batch_size': BATCH, 
          'n_channels': 1,
          'shuffle': True}

params_val = {'dim': (32, 32, 32),
          'ydim': 1,
          'batch_size': 1,
          'n_channels': 1,
          'shuffle': False}

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params_val)

# Callbacks 
csv_logger = CSVLogger(f'CNN_adattata/log_{NAME}.csv', append=True, separator=';')
tensorboard = TensorBoard(log_dir=f'CNN_adattata/logs/{NAME}')
earlystopping = EarlyStopping(monitor='val_loss', patience=10,)
mcp_save = ModelCheckpoint(f'{NAME}.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# def R2(y_true, y_pred):
#     SS_res =  K.sum(K.square(y_true - y_pred))
#     SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
#     return (1 - SS_res / (SS_tot) + K.epsilon())
# reduceLRonplateau = ReduceLROnPlateau(
#     monitor="val_loss",
#     factor=0.005,
#     patience=5,
#     verbose=0,
#     mode="auto",
#     min_delta=0.0001,
#     cooldown=0,
#     min_lr=0,
# )

cnn = Sequential()
cnn.add(
    Conv3D(8, 3, strides=1, padding='valid', activation='elu',
           data_format='channels_last', input_shape=(32, 32, 32, 1)))

cnn.add(Conv3D(16, 3, strides=1, padding='valid', activation='elu'))
cnn.add(AveragePooling3D(2))

cnn.add(Conv3D(32, 3, strides=1, padding='valid', activation='elu'))
cnn.add(AveragePooling3D(2))

cnn.add(Flatten())

cnn.add(Dense(units=16, activation='elu'))
cnn.add(Dense(units=8, activation='elu'))
cnn.add(Dense(units=8, activation='elu'))
cnn.add(Dense(units=4, activation='elu'))

cnn.add(Dense(units=1))

cnn.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mae'])
cnn.summary()

cnn.fit(x=training_generator, validation_data=validation_generator, epochs=EPOCHS, callbacks=[csv_logger, tensorboard, earlystopping, mcp_save])

score = cnn.evaluate(validation_generator)
print(f'Validation Loss: {score[0]}\n')
print(f'\nValidation MAE: {score[1]}') 
#  Returns an array which have a shape of (no_data, 1) containing prediction on val_generator
val_predictions = cnn.predict(validation_generator)


print(f'First ten theoretical outputs:\n')
for i in range(10):
	print(outputs_val[i])

print(f'\nFirst ten predictions:\n')
for i in range(10):
	print(val_predictions[i]) 
R2_val = r2_score(outputs_val, val_predictions)
print(f'R2_val = {R2_val}')
print(f"Duration: {datetime.now() - start_time}")

# Save model
cnn.save(NAME)
