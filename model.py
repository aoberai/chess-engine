import random
import datetime
import numpy as np
import tensorflow as tf



input_shape = (8, 8, 6) # Board Dimensions are 8 x 8
output_nodes = 1 # output is singular number from -1 to 1 evaluating position

''' VGG '''
# inputs = tf.keras.Input(shape=input_shape, name = 'input')
#
# conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name ='conv1_1')(inputs)
# conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name ='conv1_2')(conv1)
# # pool1 = tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_1')(conv1)
#
# conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name ='conv2_1')(conv1)
# conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name ='conv2_2')(conv2)
# # pool2 = tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_2')(conv2)
#
# conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name ='conv3_1')(conv2)
# conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name ='conv3_2')(conv3)
# conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name ='conv3_3')(conv3)
# # pool3 = tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_3')(conv3)
#
# conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv4_1')(conv3)
# conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv4_2')(conv4)
# conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv4_3')(conv4)
# # pool4 = tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), padding='same', name = 'pool_4')(conv4)
#
# conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv5_1')(conv4)
# conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv5_2')(conv5)
# conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv5_3')(conv5)
# # pool5 = tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), padding='same', name = 'pool_5')(conv5)
# flattened_input_position = tf.keras.layers.Flatten()(conv5)
#
# input_color = tf.keras.Input(shape=(1,))
# combined_input = tf.keras.layers.Concatenate()([flattened_input_position, input_color])
# # first_conv = tf.keras.layers.Conv2D(64, (3, 3), input_shape=input_shape)(combined_input)
# # flatten = tf.keras.layers.Flatten(input_shape=input_shape)(first_conv)
# # first_dense = tf.keras.layers.Dense(units=128, activation='relu')(combined_input)
# second_hidden_dense = tf.keras.layers.Dense(units=64, activation='relu')(combined_input)
# # dropout = tf.keras.layers.Dropout(0.2)(second_hidden_dense)
# output_dense = tf.keras.layers.Dense(units=output_nodes, activation='tanh')(second_hidden_dense)
# model = tf.keras.Model(inputs=[inputs, input_color], outputs=output_dense)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)

''' Custom Model '''

with strategy.scope():
    input_position = tf.keras.Input(shape=input_shape)
    input_position_conv = tf.keras.layers.Conv2D(16, (7, 7), padding="same", activation='relu')(input_position)
    input_position_conv2 = tf.keras.layers.Conv2D(16, (7, 7), padding="same", activation='relu')(input_position_conv)
    input_position_conv3 = tf.keras.layers.Conv2D(16, (5, 5), padding="same", activation='relu')(input_position_conv2)
    input_position_conv4 = tf.keras.layers.Conv2D(16, (5, 5), padding="same", activation='relu')(input_position_conv3)
    input_position_conv5 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu')(input_position_conv4)
    input_position_conv6 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu')(input_position_conv5)
    input_position_conv7 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu')(input_position_conv6)
    input_position_conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu')(input_position_conv7)
    input_position_conv9 = tf.keras.layers.Conv2D(128, (2, 2), padding="same", activation='relu')(input_position_conv8)
    input_position_conv_final = tf.keras.layers.Conv2D(128, (2, 2), activation='relu')(input_position_conv9)
    # input_position_conv8 = tf.keras.layers.Conv2D(512, (1, 1), activation='sigmoid')(input_position_conv7)
    flattened_input_position = tf.keras.layers.Flatten()(input_position_conv_final)
    # second_hidden_dense = tf.keras.layers.Dense(units=128, activation='relu')(flattened_input_position)
    output_dense = tf.keras.layers.Dense(units=output_nodes, activation='tanh')(flattened_input_position)
    model = tf.keras.Model(inputs=input_position, outputs=output_dense)

    model.summary()
    tf.keras.utils.plot_model(model, to_file="assets/model.png", show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['MSE'])

# memmap the file
evaluations_train_memmap = np.load('evaluations_3M.npy', mmap_mode='r')
positions_train_memmap = np.load('positions_3M.npy', mmap_mode='r')

# evaluations_val_memmap = np.load('evaluations_val100K.npy', mmap_mode='r')
# positions_val_memmap = np.load('positions_val100K.npy', mmap_mode='r')

assert len(evaluations_train_memmap) == len(positions_train_memmap)



training_data_seen_indexes = []
def training_data_generator():
    for i in range(0, len(positions_train_memmap)):
        yield (positions_train_memmap[i], evaluations_train_memmap[i])
        # return (iter(positions_train_memmap), iter(evaluations_train_memmap))

''' Shows generator working '''
# print("Running generator")
# generator = training_data_generator()
# count = 0
# while True:
#     try:
#         print(next(generator))
#         count+=1
#         print(count)
#     except Exception as e:
#         print(e)
#         break
# print(count)


total_dataset = tf.data.Dataset.from_generator(
        generator=training_data_generator, output_signature=(tf.TensorSpec(shape=input_shape, dtype=np.int32), tf.TensorSpec(shape=(), dtype=np.float32)))
total_dataset.cache()

# val_dataset = tf.data.Dataset.from_generator(
        # generator=validation_data_generator, output_signature=(tf.TensorSpec(shape=input_shape, dtype=np.int32), tf.TensorSpec(shape=(), dtype=np.float32)))

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# total_dataset = total_dataset.shuffle(2000000)
validation_set_size = 100000
batch_size = 256
val_dataset = total_dataset.take(validation_set_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_dataset = total_dataset.skip(validation_set_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=6)

# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_dataset = train_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)

model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[early_stopping_callback, tensorboard_callback], use_multiprocessing=True, workers=8)


model.save("chess_engine_vlatest.h5")

