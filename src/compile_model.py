from tensorflow.keras import layers, models

def compile_model(architecture: str, input_shape):
    model = models.Sequential()
    first_layer = True
    for layer in architecture.split("->"):
        layer = layer.strip()
        if layer.startswith("Dense"):
            params = layer[len("Dense("):-1]
            units, activation = params.split(",")
            units = int(units)
            activation = activation.strip()
            if first_layer:
                model.add(layers.Dense(units, activation=activation, input_shape=input_shape))
                first_layer = False
            else:
                model.add(layers.Dense(units, activation=activation))
        elif layer.startswith("Conv2D"):
            params = layer[len("Conv2D("):-1]
            filters, kernel, activation = params.split(",")
            model.add(layers.Conv2D(int(filters), int(kernel), activation=activation.strip(), padding="same"))
        elif layer.startswith("MaxPool"):
            size = int(layer[len("MaxPool("):-1])
            model.add(layers.MaxPooling2D(pool_size=(size, size)))
        elif layer.startswith("Flatten"):
            model.add(layers.Flatten())
    return model