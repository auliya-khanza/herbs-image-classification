import io
import json

import torch as pt
import torchvision.transforms as transforms

from PIL import Image

from flask import Flask, jsonify, request
from flask_cors import CORS

class_index = {
    0: "JAHE",
    1: "KUNYIT",
    2: "LENGKUAS"
}
model = pt.load("modelkita_v3.pth", map_location=pt.device("cpu"))
model.eval()


def transform_image(image_bytes):
    test_transform = transforms.Compose([
        transforms.Resize(500),          
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return test_transform(image)


def get_prediction(image_bytes):
    tensor_img = transform_image(image_bytes=image_bytes)
    outputs = model(tensor_img.unsqueeze(0))
    predicted_idx = outputs.max(1).indices

    return class_index[predicted_idx.item()]


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'hello':'test success'})

if __name__ == '__main__':
    app.run()