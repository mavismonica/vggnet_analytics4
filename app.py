from flask import Flask, render_template, request
from numpy.lib.type_check import imag
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os

#create flask instance
app = Flask(__name__)

#set max size of file as 10MB
# app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#allow file with .png, .jpg and .jpeg
ALLOWED_EXTENSIONS = ['png','jpg','jpeg']
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.compat.v1.get_default_graph()

def read_image(filename):
    img = load_img(filename,color_mode = "grayscale",target_size=(250,250)) 
    img = img_to_array(img)
    img = img.reshape(1,250,250,1)
    img = imag.astype('float32')
    img = img / 255.0
    return img

#get - request data from the web server, post- send data to the web server
@app.route("/", methods = ['GET','POST'])
def home():
    return render_template('home.html')


@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        # try:
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('C:/Users/mavis/Documents/ana4/static/images',filename)
            file.save(file_path)
            img = read_image(file_path)

            with graph.as_default():
                model1 = load_model('C:/Users/mavis/Documents/ana4/trained_VGG_model.h5')
                print(model1.head())
                class_prediction = model1.predict(img)
                print(class_prediction)

            if class_prediction[0] == 0:
                product = "BEANS"
            elif class_prediction[0] == 1:
                product = "CAKE"
            elif class_prediction[0] == 2:
                product = "CANDY"
            elif class_prediction[0] == 3:
                product = "CEREAL"
            elif class_prediction[0] == 4:
                product = "CHIPS"
            elif class_prediction[0] == 5:
                product = "CHOCOLATE"
            elif class_prediction[0] == 6:
                product = "COFFEE"
            elif class_prediction[0] == 7:
                product = "CORN"
            elif class_prediction[0] == 8:
                product = "FISH"
            elif class_prediction[0] == 9:
                product = "FLOUR"
            elif class_prediction[0] == 10:
                product = "HONEY"
            elif class_prediction[0] == 11:
                product = "JAM"
            elif class_prediction[0] == 12:
                product = "JUICE"
            elif class_prediction[0] == 13:
                product = "MILK"
            elif class_prediction[0] == 14:
                product = "NUTS"
            elif class_prediction[0] == 15:
                product = "OIL"
            elif class_prediction[0] == 16:
                product = "PASTA"
            elif class_prediction[0] == 17:
                product = "RICE"
            elif class_prediction[0] == 18:
                product = "SODA"
            elif class_prediction[0] == 19:
                product = "SPICES"
            elif class_prediction[0] == 20:
                product = "SUGAR"
            elif class_prediction[0] == 21:
                product = "TEA"
            elif class_prediction[0] == 22:
                product = "TOMATO_SAUCE"
            elif class_prediction[0] == 23:
                product = "VINEGAR"
            else:
                product = "WATER"

            return render_template('predict.html', product = product, user_image = file_path)

        # except Exception as e:
        #     return "Unable t read the file, please check file extension."

    return render_template('predict.html')     

if __name__ == "__main__":
    init()
    app.run() 