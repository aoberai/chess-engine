import numpy as np
import tensorflow as tf

input_shape = (8, 8, 1) # Board Dimensions are 8 x 8
output_nodes = 1 # output is singular number from -1 to 1 evaluating position

input_position = tf.keras.Input(shape=(8, 8, 1))
input_position_conv = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(input_position)
input_position_conv2 = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(input_position_conv)
input_position_conv3 = tf.keras.layers.Conv2D(32, (3, 3), strides=2)(input_position_conv2)
input_position_conv4 = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(input_position_conv3)
input_position_conv5 = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(input_position_conv4)
input_position_conv6 = tf.keras.layers.Conv2D(128, (1, 1))(input_position_conv5)
input_position_conv7 = tf.keras.layers.Conv2D(128, (1, 1))(input_position_conv6)

flattened_input_position = tf.keras.layers.Flatten()(input_position_conv7)

input_color = tf.keras.Input(shape=(1,))
combined_input = tf.keras.layers.Concatenate()([flattened_input_position, input_color])
# first_conv = tf.keras.layers.Conv2D(64, (3, 3), input_shape=input_shape)(combined_input)
# flatten = tf.keras.layers.Flatten(input_shape=input_shape)(first_conv)
# first_dense = tf.keras.layers.Dense(units=128, activation='relu')(combined_input)
second_hidden_dense = tf.keras.layers.Dense(units=64, activation='relu')(combined_input)
# dropout = tf.keras.layers.Dropout(0.2)(second_hidden_dense)
output_dense = tf.keras.layers.Dense(units=output_nodes, activation='tanh')(second_hidden_dense)
model = tf.keras.Model(inputs=[input_position, input_color], outputs=output_dense)
model.summary()
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy', 'MSE'])
evaluations = np.load('evaluations_200.npy')
positions = np.load('positions_200.npy')
colors = np.load('colors_200.npy')
np.random.seed(0)
np.random.shuffle(evaluations)
np.random.seed(0)
np.random.shuffle(positions)
# for i in range(0, len(evaluations)):
#     print(positions[i])
#     print("Evaluation", evaluations[i])
#     print("Color", "White" if colors[i] == 1 else "Black")
print(len(evaluations))
print(evaluations[1])
print(positions[1])
print(colors[1])
model.fit(x=[positions, colors], y=evaluations, epochs=150, validation_split=0.2)





