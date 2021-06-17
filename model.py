import datetime
import numpy as np
import tensorflow as tf


piece_char_2_int = {
        'p' : (1,0,0,0,0,0,0,0,0,0,0,0),
        'P' : (0,0,0,0,0,0,1,0,0,0,0,0),
        'n' : (0,1,0,0,0,0,0,0,0,0,0,0),
        'N' : (0,0,0,0,0,0,0,1,0,0,0,0),
        'b' : (0,0,1,0,0,0,0,0,0,0,0,0),
        'B' : (0,0,0,0,0,0,0,0,1,0,0,0),
        'r' : (0,0,0,1,0,0,0,0,0,0,0,0),
        'R' : (0,0,0,0,0,0,0,0,0,1,0,0),
        'q' : (0,0,0,0,1,0,0,0,0,0,0,0),
        'Q' : (0,0,0,0,0,0,0,0,0,0,1,0),
        'k' : (0,0,0,0,0,1,0,0,0,0,0,0),
        'K' : (0,0,0,0,0,0,0,0,0,0,0,1),
        '.' : (0,0,0,0,0,0,0,0,0,0,0,0),
}


int_2_piece_char = {v: k for k, v in piece_char_2_int.items()}


input_shape = (8, 8, 12) # Board Dimensions are 8 x 8
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
#
''' Custom Model '''
input_position = tf.keras.Input(shape=input_shape)
input_position_conv = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation='relu')(input_position)
input_position_conv2 = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation='relu')(input_position_conv)
input_position_conv3 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu')(input_position_conv2)
input_position_conv4 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu')(input_position_conv3)
input_position_conv5 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='sigmoid')(input_position_conv4)
input_position_conv6 = tf.keras.layers.Conv2D(128, (1, 1), activation='sigmoid')(input_position_conv5)
input_position_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='sigmoid')(input_position_conv6)

flattened_input_position = tf.keras.layers.Flatten()(input_position_conv7)

# first_conv = tf.keras.layers.Conv2D(64, (3, 3), input_shape=input_shape)(combined_input)
# flatten = tf.keras.layers.Flatten(input_shape=input_shape)(first_conv)
# first_dense = tf.keras.layers.Dense(units=128, activation='relu')(combined_input)
second_hidden_dense = tf.keras.layers.Dense(units=16, activation='tanh')(flattened_input_position)
# dropout = tf.keras.layers.Dropout(0.2)(second_hidden_dense)
output_dense = tf.keras.layers.Dense(units=output_nodes, activation='tanh')(second_hidden_dense)
model = tf.keras.Model(inputs=input_position, outputs=output_dense)

model.summary()
tf.keras.utils.plot_model(model, to_file="assets/model.png", show_shapes=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['MSE'])
evaluations = np.load('evaluations_2.25M.npy')
positions = np.load('positions_2.25M.npy')
# colors = np.load('colors_W750K.npy')

# np.random.seed(0)
# np.random.shuffle(evaluations)
# np.random.seed(0)
# np.random.shuffle(positions)


assert len(evaluations) == len(positions)


# Checks that all positions and evaluations are correctly associated after scrambling - successful 
# print(evaluations[np.argmax(evaluations)])
# for i in range(0, len(positions[np.argmax(evaluations)])):
#     for j in range(0, len(positions[np.argmax(evaluations)][i])):
#         print(" ", int_2_piece_char[tuple(positions[np.argmax(evaluations)][i][j].tolist())], end = ' ')
#     print("\n")
#

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=positions, y=evaluations, epochs=35, validation_split=0.2, batch_size = 64, shuffle=True, callbacks=[tensorboard_callback])


if input("Do you want to save model? y for yes, n for no?\n") == 'y':
    model.save("chess_engine_%s.h5" % input("Enter Model Name: "))





