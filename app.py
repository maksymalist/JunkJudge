from utils import *
import torch
from PIL import Image
from gen_dataset import preds_to_data, data_to_preds
from algo import final_say
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
import urllib.request
import os
from io import BytesIO
import base64

seed_everything()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def index():
    return "Hello, World!"

@app.route('/api/v1/predict', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def predict():
    json_data = request.get_json() 
    b64 = json_data['image_b64']
        
    image = Image.open(BytesIO(base64.b64decode(b64)))
    c1, c2, v1, v2, out1, out2 = get_predictions(image)

    probas = preds_to_data(c1, c2).unsqueeze(0).to(DEVICE)
    prediction = Morpheus(probas).argmax(1).item()
    out3 = list(CLASSES_1.keys())[prediction]
    
    final_verdict = final_say(v1, v2, out1, out2, out3, probas)
    print(final_verdict)
    
    return jsonify({
        "result": final_verdict,
        "m1_confidence": v1,
        "m2_confidence": v2,
        "probabilities": probas.tolist(),
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# VISUALIZATION

# fig, ax = plt.subplots(1, 3, figsize=(10, 10))

# ax[0].pie(v1.values(), labels=v1.keys())
# ax[0].margins(x=20, y=10)
# ax[0].set_title("First Model")

# ax[1].pie(v2.values(), labels=v2.keys())
# ax[1].margins(x=20, y=10)
# ax[1].set_title("Second Model")

# ax[2].imshow(img1)
# ax[2].axis("off")
# ax[2].margins(x=20, y=10)
# ax[2].set_title(final_verdict)

# plt.show()