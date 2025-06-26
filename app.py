from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(_name_)
model = load_model('vgg16_model.h5')

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Butterfly class names (you can change these based on your model)
class_names = ['Monarch Butterfly', 'Swallowtail', 'Painted Lady', 'Blue Morpho', 'Common Jezebel']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded."

    file = request.files['image']
    if file.filename == '':
        return "No file selected."

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        predicted_label = class_names[class_index]

        return render_template('output.html', prediction=predicted_label, image_path=filepath)

    return "Something went wrong!"

if _name_ == '_main_':
    app.run(debug=True)
