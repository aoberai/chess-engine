import numpy as np
import tensorflow as tf

input_shape = (8, 8, 1) # Board Dimensions are 8 x 8
output_nodes = 1 # output is singular number from -1 to 1 evaluating position

input_position = tf.keras.Input(shape=(8, 8, 1))
flattened_input_position = tf.keras.layers.Flatten()(input_position)
input_color = tf.keras.Input(shape=(1,))
combined_input = tf.keras.layers.Concatenate()([flattened_input_position, input_color])
first_dense = tf.keras.layers.Dense(units=64)(combined_input)
hidden_dense = tf.keras.layers.Dense(units=32)(first_dense)
dropout = tf.keras.layers.Dropout(0.2)(hidden_dense)
output_dense = tf.keras.layers.Dense(units=output_nodes)(dropout)
model = tf.keras.Model(inputs=[input_position, input_color], outputs=output_dense)
model.summary()

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# evaluations = np.load('evaluations.npy')
# positions = np.load('positions.npy')
# np.random.seed(0)
# np.random.shuffle(evaluations)
# np.random.seed(0)
# np.random.shuffle(positions)
# for i in range(0, len(evaluations)):
#     print(positions[i])
#     print("Evaluation", evaluations[i])
# # model.fit(x=positions, y=evaluations, epochs=8, validation_split = 0.2)
#
#
#


