import tensorflow as tf
input_shape = (8, 8, 1) # Board Dimensions are 8 x 8
output_nodes = 1 # output is singular number from -1 to 1 evaluating position
model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=64, input_shape=input_shape), tf.keras.layers.Dense(units=128), tf.keras.Dropout(0.2), tf.keras.layers.Dense(units=output_nodes)])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])




