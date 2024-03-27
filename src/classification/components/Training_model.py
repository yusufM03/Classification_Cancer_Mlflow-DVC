import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from classification.config.configuration import *
import os

#Component 
class training_Model:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )


    def Processing_Data(self):
        Train_Data=os.path.join(self.config.training_data,'Train')
        X_train=[]
        y_train=[]
        for classe in os.listdir(Train_Data):
              
          for img in os.listdir(os.path.join(Train_Data,classe)) :
            img_path=os.path.join(Train_Data,classe,img)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(self.config.params_image_size[0], self.config.params_image_size[1]))
            x = tf.keras.preprocessing.image.img_to_array(img)

            x = tf.keras.applications.vgg16.preprocess_input(x)
            X_train.append(x)
            y_train.append(classe)
        

      
  
        val_Data=os.path.join(self.config.training_data,'valid')
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


                # Convert y_train to a NumPy array
        y_train_array = np.array(y_train)
        y_val_array = np.array(y_val)

        # Reshape y_train for one-hot encoding
        y_train_reshaped = y_train_array.reshape(-1, 1)
        y_val_reshaped = y_val_array.reshape(-1, 1)

        # Initialize the one-hot encoder
        onehot_encoder = OneHotEncoder(sparse=False)

        # Fit one-hot encoder and transform y_train
        y_train_onehot = onehot_encoder.fit_transform(y_train_reshaped)
        y_val_onehot = onehot_encoder.transform(y_val_reshaped)
        

        
        X_train=np.array(X_train)

        X_val=np.array(X_val)
        self.X_train=X_train
        self.y_train=  y_train_onehot
        self.X_val=X_val
        self.y_val=   y_val_onehot
      
  
  

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        
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
    
        self.model.fit(
            self.X_train,self.y_train,
            epochs=self.config.params_epochs,
            validation_data=(self.X_val,self.y_val),
            batch_size=self.config.params_batch_size
        )
        
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        os.mkdir("Model")
        os.rename(self.config.trained_model_path,"Model")


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
