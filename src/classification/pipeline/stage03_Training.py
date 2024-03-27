

from classification.config.configuration import *
from classification.components.Training_model import *
from classification import logger 


stage_name="Training Model"

class TrainignModelPipeleine:
    def __init__(self) :
        pass
    def main(self):
      try:
          config = configurationManager()
          Training_model_config = config.get_training_config()
          Training_model = training_Model(config=Training_model_config)
          Training_model.get_base_model()
          
          Training_model.Processing_Data()
          Training_model.train()
      except Exception as e:
          raise e
            

if __name__=='__main__':
    try :
          logger.info(f">>>>>>>Stage{stage_name} started <<<<<<<<<<<")
          obj=TrainignModelPipeleine()
          obj.main()
          logger.info(f">>>>>>>Stage{stage_name} completed <<<<<<<<<<<")
    except Exception as e :
         logger.exception(e)
         raise e
    

