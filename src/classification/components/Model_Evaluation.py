import os
import tensorflow as tf
from classification import logger
import mlflow
import mlflow.keras
import numpy as np
from classification.config.configuration import *
from classification.utils.common import save_json
from classification.entity.config_entity import EvaluationConfig
#Component 

from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder

class Evaluate_Model:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def Processing_Data(self):

      
  
        val_Data=os.path.join(self.config.training_data,'test')
        X_val=[]
        y_val=[]

        for classe in os.listdir(val_Data):
            
          for img in os.listdir(os.path.join(val_Data,classe)) :
            img_path=os.path.join(val_Data,classe,img)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(self.config.params_image_size[0], self.config.params_image_size[0]))
            x = tf.keras.preprocessing.image.img_to_array(img)

            x = tf.keras.applications.vgg16.preprocess_input(x)
            X_val.append(x)
            y_val.append(classe)

        y_val_array = np.array(y_val)
        y_val_reshaped = y_val_array.reshape(-1, 1)
      

        # Initialize the one-hot encoder
        onehot_encoder = OneHotEncoder(sparse=False)

        y_val_onehot =onehot_encoder.fit_transform(y_val_reshaped)

        x_val=np.array(X_val)

        self.val=x_val
        self.y_val=y_val_onehot
      
      
        

    

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)


    def evaluation(self):
        self.model = self.load_model(self.config.trained_model_path)

        self.score = self.model.evaluate(self.val,self.y_val)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)


  
    def log_into_mlflow(self):
      os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/yusufM03/Classification_Mlflow-DVC.mlflow"
      os.environ["MLFLOW_TRACKING_USERNAME"]="yusufM03"
      os.environ["MLFLOW_TRACKING_PASSWORD"]="3735ecba8f2f0dccee05c421728f3ed8abd06c96"
      
      mlflow.set_registry_uri(self.config.MLflow_URI)
      tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
      
      with mlflow.start_run():
          mlflow.log_params(self.config.all_params)
          mlflow.log_metrics(
              {"loss": self.score[0], "accuracy": self.score[1]}
          )
          # Model registry does not work with file store
          if tracking_url_type_store != "file":

              # Register the model
              # There are other ways to use the Model Registry, which depends on the use case,
              # please refer to the doc for more information:
              # https://mlflow.org/docs/latest/model-registry.html#api-workflow
              mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
          else:
              mlflow.keras.log_model(self.model, "model")
    