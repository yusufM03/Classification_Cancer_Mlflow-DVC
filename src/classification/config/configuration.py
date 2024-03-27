
from classification.constants import *
from classification.utils.common import read_yaml,create_directories
from classification.entity.config_entity import *


import os 

class configurationManager:
  def __init__(
    self,
    configfile_path=CONFIG_FILE_PATH,
    paramsfile_path=PARAMS_FILE_PATH) :
  
    self.config=read_yaml(configfile_path)
    self.params=read_yaml(paramsfile_path)
    create_directories([self.config.artifacts_root])
  
  def Evaluation_config(self) -> EvaluationConfig:
      Evaluation= self.config.Evaluation
      params = self.params
      training_data = os.path.join(self.config.data_ingestion.unzip_dir,"Data")
      create_directories([
          Path(Evaluation.root_dir)
      ])
      


      Evaluation_config = EvaluationConfig(
          root_dir=Path(Evaluation.root_dir),
          trained_model_path=Path(Evaluation.trained_model_path),
          MLflow_URI=Evaluation.MLflow_URI,
          training_data=Path(training_data),

          all_params=params,
          params_batch_size=params.BATCH_SIZE,
          params_image_size=params.IMAGE_SIZE
      )
      return Evaluation_config





  def data_ingestion_configuration(self) ->DataIngestionConfig:
    config=self.config.data_ingestion
    create_directories([self.config.artifacts_root])
    data_ingestion_config=DataIngestionConfig(
      
    root_dir= config.root_dir,
    source_URL= config.source_URL,
    local_data_file=config.local_data_file,
    unzip_dir=config.unzip_dir

      
      
    )

    return data_ingestion_config
  
  def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
      config = self.config.prepare_base_model
      
      create_directories([config.root_dir])

      prepare_base_model_config = PrepareBaseModelConfig(
          root_dir=Path(config.root_dir),
          base_model_path=Path(config.base_model_path),
          updated_base_model_path=Path(config.updated_base_model_path),
          params_image_size=self.params.IMAGE_SIZE,
          params_learning_rate=self.params.LEARNING_RATE,
          params_include_top=self.params.INCLUDE_TOP,
          params_weights=self.params.WEIGHTS,
          params_classes=self.params.CLASSES
      )

      return prepare_base_model_config
  
  def get_training_config(self) -> TrainingConfig:
      training = self.config.training
      prepare_base_model = self.config.prepare_base_model
      params = self.params
      training_data = os.path.join(self.config.data_ingestion.unzip_dir,"Data")
      create_directories([
          Path(training.root_dir)
      ])


      training_config = TrainingConfig(
          root_dir=Path(training.root_dir),
          trained_model_path=Path(training.trained_model_path),
          updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
          training_data=Path(training_data),
          params_epochs=params.EPOCHS,
          params_batch_size=params.BATCH_SIZE,
          params_is_augmentation=params.AUGMENTATION,
          params_image_size=params.IMAGE_SIZE,
          classes=params.CLASSES
      )
      return training_config
  