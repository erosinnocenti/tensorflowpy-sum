import tensorflow as tf
import random
from numpy import array

def main():
    inputs = []
    outputs = []

    for i in range(100):
        for j in range(100):
            inputs.append([i, j])
            outputs.append(i + j)

    model = tf.keras.models.Sequential()

    layer1 = tf.keras.layers.Dense(1, input_shape=[2], use_bias=True)

    model.add(layer1)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001), loss=tf.losses.mean_squared_error)
    
    model.summary()

    model.fit(x=inputs, y=outputs, batch_size=10000, epochs=10, shuffle=True)

    for i in range(3):
        a = random.randint(100, 200)
        b = random.randint(100, 200)
        expected = a + b

        result = model.predict([[a, b]])

        print(a, " + ", b, " = ", expected, " --- ", result)

main()