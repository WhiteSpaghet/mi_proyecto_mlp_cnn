import os
from src.compile_model import compile_model
from src.firebase_utils import init_firebase, upload_model_to_storage, save_metrics_to_firestore
from src.train import load_mnist
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    print("🚀 Proyecto: Intérprete y Entrenamiento de Redes Neuronales (MLP/CNN)")
    print("Selecciona una opción:")
    print("1. Entrenar MLP (denso)")
    print("2. Entrenar CNN (convolucional)")
    print("3. Evaluar modelo guardado (.h5)")
    opcion = input("👉 Escribe 1, 2 o 3: ")

    if opcion == "1":
        architecture = "Dense(256,relu) -> Dense(128,relu) -> Dense(10,softmax)"
        input_shape = (784,)
        as_cnn = False
    elif opcion == "2":
        architecture = "Conv2D(32,3,relu) -> MaxPool(2) -> Conv2D(64,3,relu) -> MaxPool(2) -> Flatten() -> Dense(128,relu) -> Dense(10,softmax)"
        input_shape = (28,28,1)
        as_cnn = True
    elif opcion == "3":
        model_path = input("Ruta del modelo (.h5): ")
        model = tf.keras.models.load_model(model_path)
        _, _, x_test, y_test = load_mnist(as_cnn=('conv' in model_path.lower()))
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f"Precisión en test: {test_acc:.4f}")
        return
    else:
        print("Opción no válida.")
        return

    print("\n📦 Cargando dataset MNIST...")
    x_train, y_train, x_test, y_test = load_mnist(as_cnn=as_cnn)

    print("\n⚙️ Compilando arquitectura:")
    print(architecture)
    model = compile_model(architecture, input_shape=input_shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("\n🏋️ Entrenando modelo...")
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128)

    print("\n📊 Evaluando en conjunto de test...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"✅ Precisión final: {test_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    model_path = "models/model_trained.h5"
    model.save(model_path)
    print(f"\n💾 Modelo guardado en {model_path}")

    # Graficar resultados
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.legend()
    plt.title('Evolución de precisión')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/training_accuracy.png")
    plt.close()

    # Firebase opcional
    subir = input("\n¿Quieres subir resultados a Firebase? (s/n): ").lower()
    if subir == 's':
        sa_path = "config/serviceAccountKey.json"
        bucket_name = input("Nombre del bucket (o deja vacío): ").strip() or None
        db, bucket = init_firebase(sa_path, bucket_name)
        save_metrics_to_firestore(db, "resultados_modelo",
                                  {"accuracy": float(test_acc), "loss": float(test_loss)})
        if bucket:
            url = upload_model_to_storage(bucket, model_path, "model_trained.h5")
            print(f"📤 Modelo subido a Firebase Storage: {url}")
        else:
            print("Datos subidos a Firestore correctamente.")

    print("\n✅ Proceso completo.")

if __name__ == "__main__":
    main()