#app.py
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)


# Load trained model
model = tf.keras.models.load_model("plant_species_model.h5")

# Class labels (replace with actual class names)
class_labels = ['Plant A', 'Plant B', 'Plant C']

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", prediction="No file uploaded.")

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Preprocess the image
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_label = class_labels[np.argmax(prediction)]

    return render_template("index.html", prediction=predicted_label)

if __name__ == "__main__":
    app.run(debug=True)

