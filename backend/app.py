from flask import Flask
from flask import request
from flask import jsonify
from flask import send_file
from flask import send_from_directory
from flask_cors import CORS

import cv2
import numpy as np
from keras.models import load_model
import pandas as pd

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = load_model('model.h5',  compile=False)

def allowed_file(filename):
  extension = filename.rsplit('.', 1)[1].lower()
  return '.' in filename and extension in ALLOWED_EXTENSIONS

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
labels = pd.read_csv('label.csv')

@app.route('/', methods=['POST'])
def home():
    if 'img' not in request.files:
        return jsonify({
            'message': 'Missing file'
        }), 400
    file = request.files['img']
    if file.filename == '':
        return jsonify({
            'message': 'No file selected'
        }), 400
    
    if file and allowed_file(file.filename):
        try:
            filestr = file.read()
            img = np.frombuffer(filestr, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = np.asarray(img)
            img = cv2.resize(img, (32, 32))
            img = preprocessing(img)
            img = img.reshape(1, 32, 32, 1)
            pred = model.predict(img)
            pred = np.argmax(np.round(pred), axis=1)
            print(pred)
            return jsonify({
                'message': 'Success',
                'traffic_id': str(pred[0]),
                'traffic_name': labels.iloc[[pred[0]]].to_json()
            })
        except Exception as e:
            print(e)
            return jsonify({
                'message': 'Server error'
            }), 500
    else:
      return jsonify({
        'message': 'File doesnt support'
      }), 400
@app.route('/model/<string:filename>')
def get_file(filename):
    return send_file('tfjs/{}'.format(filename))
print("Load ok")
app.run(host="0.0.0.0")
from flask import Flask, request
app = Flask(__name__)


