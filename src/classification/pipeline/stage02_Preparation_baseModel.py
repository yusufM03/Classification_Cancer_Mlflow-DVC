


from classification.components.Model_Preparation import *
from classification import logger

stage_name="Preparation Base Model"

class PreparationBaseModelPipeleine:
    def __init__(self) :
        pass
    def main(self):
              
              
        try:
            config = configurationManager()
            prepare_base_model_config = config.get_prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
        except Exception as e:
            raise e

      

if __name__=='__main__':
    try :
          logger.info(f">>>>>>>Stage{stage_name} started <<<<<<<<<<<")
          obj=PreparationBaseModelPipeleine()
          obj.main()
          logger.info(f">>>>>>>Stage{stage_name} completed <<<<<<<<<<<")
    except Exception as e :
         logger.exception(e)
         raise e
    