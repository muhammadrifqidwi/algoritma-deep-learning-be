import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array

DEFAULT_MODEL_PATH = os.path.join("models", "object_cnn_model.h5")

class ObjectCNN:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, img_size=(64, 64)):
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.class_names = ["Cat", "Dog"]
        self.unknown_threshold = 0.50

        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=(64, 64, 3)),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(2, activation="softmax")
        ])

        model.compile(
            optimizer=Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def load_cifar10_cat_dog(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        train_idx = np.where((y_train == 3) | (y_train == 5))[0]
        test_idx = np.where((y_test == 3) | (y_test == 5))[0]

        x_train, y_train = x_train[train_idx], y_train[train_idx]
        x_test, y_test = x_test[test_idx], y_test[test_idx]

        y_train = (y_train == 5).astype(int)
        y_test = (y_test == 5).astype(int)

        x_train = np.array([np.array(Image.fromarray(img).resize(self.img_size)) for img in x_train])
        x_test = np.array([np.array(Image.fromarray(img).resize(self.img_size)) for img in x_test])

        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)

        return x_train, x_test, y_train, y_test

    def train_cifar10(self, epochs=25, batch_size=32):
        print("📥 Loading CIFAR-10 (Cat & Dog)...")
        x_train, x_test, y_train, y_test = self.load_cifar10_cat_dog()

        print("🚀 Building lightweight model...")
        model = self.build_model()

        print("🏋 Training...")
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs=epochs, batch_size=batch_size, verbose=1)

        model.save(self.model_path)
        self.model = model

        print("✅ Model saved:", self.model_path)

    def load(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            self.model = models.load_model(self.model_path)
        return self.model

    def _prepare_image(self, image_input):
        if isinstance(image_input, str) and os.path.exists(image_input):
            img = Image.open(image_input).convert("RGB")
        else:
            image_input.seek(0)
            img = Image.open(image_input).convert("RGB")

        img = img.resize(self.img_size)
        arr = img_to_array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, image_input):
        model = self.load()
        arr = self._prepare_image(image_input)

        preds = model.predict(arr)
        prob = float(np.max(preds[0]))
        idx = int(np.argmax(preds[0]))

        if prob < self.unknown_threshold:
            return "Unknown", prob

        return self.class_names[idx], prob
