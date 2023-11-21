from flask import Flask, request, jsonify
import numpy as np
from model import cv_image_classify

app = Flask(__name__)


@app.route("/")
def index():
    return '''
    <h1>Welcome to Image Classifier ML Service!</h1>
    <p><b><i>Developed by Mohamad Oghli<i></b></p>
    '''


@app.route('/image_classify', methods=['POST'])
def ic_inference():
    data = request.get_json()
    if 'image' in data:
        req_image = np.array(data['image'], dtype="uint8")
        image_class = cv_image_classify(req_image)
        return jsonify({"img_class": image_class})
    return jsonify({"message": "Sorry, Invalid Image Parameter!"})


if __name__ == "__main__":
    app.run()
