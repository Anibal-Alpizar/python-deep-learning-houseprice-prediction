import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# meters of the house
meters = np.array([50, 100, 150, 200, 250, 300,
                  350, 400, 450, 500], dtype=float)
# price of the house
price = np.array([500, 1000, 1500, 2000, 2500, 3000,
                 3500, 4000, 4500, 5000], dtype=float)


# model
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(
    0.1), loss='mean_squared_error')

# training the model
print('training the model')
history = model.fit(meters, price, epochs=1000, verbose=False)
print('model trained')

plt.xlabel('Epoch number')
plt.ylabel('Loss magnitude')
plt.plot(history.history['loss'])
plt.show()


input_meters = float(input('Enter the meters of the house: '))

result = model.predict([input_meters])

print('A house with' + str(input_meters) +
      ' meters, costs about ' + str(result) + ' dollars')
