from flask import Flask, request, jsonify, render_template
import os
#from flask_cors import CORS, cross_origin
#from src.classification.utils.common import decodeImage
from classification.pipeline.prediction import PredictionPipeline
import tensorflow as tf
import numpy as np
UPLOAD_FOLDER = os.path.join('static', 'upload')
# Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__ ,static_folder='static', static_url_path='/static' , template_folder='templates' )

captureimg=None

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

@app.route("/", methods=['GET'])
def home():
    return render_template('Prediction_system.html')



@app.route("/upload", methods=['POST'])
def upload():
    # Get the file from the request
    file = request.files['image']

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    uploaded_image_path = os.path.join("static", "upload", file.filename)


    
    predicted_label=PredictionPipeline(filename=uploaded_image_path).predict()

    # Return the prediction result and the image data or URL
    return jsonify({"pred": predicted_label, "imageUrl": f"/static/upload/{file.filename}"})




if __name__ == '__main__':

    app.run(debug=True)