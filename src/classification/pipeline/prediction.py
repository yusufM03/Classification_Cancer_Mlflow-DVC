import numpy as np
import tensorflow as tf 
import os
from classification.config.configuration import *
from flask import  jsonify

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
        

    
    def predict(self):
      ## load model
      print('encours')
        
      model = tf.keras.models.load_model(os.path.join("model", "model.h5"))

      imagename = self.filename
      img = tf.keras.preprocessing.image.load_img(imagename, target_size=(224, 224))
      test_image = tf.keras.preprocessing.image.img_to_array(img)

      test_image = tf.keras.applications.vgg16.preprocess_input(test_image)
      test_image = np.expand_dims(test_image, axis=0)
      result = model.predict(test_image)
      predicted_class = np.argmax(result[0])
      classes={0:'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',1:'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',2:'normal',3:'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'}
      predicted_label = classes[predicted_class]

      # Return the prediction result and the image data or URL
      return predicted_label