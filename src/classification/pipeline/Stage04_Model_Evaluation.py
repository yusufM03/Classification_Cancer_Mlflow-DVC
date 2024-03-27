



from classification.config.configuration import configurationManager
from classification.components.Model_Evaluation import *
from classification import logger 


stage_name="Evaluate Model"

class ModelEvaluationPipeleine:
    def __init__(self) :
        pass
    def main(self):
      try:
        config = configurationManager()
        Evaluate_model_config = config.Evaluation_config()
        Evaluate_model = Evaluate_Model(config=Evaluate_model_config )
        Evaluate_model.load_model("artifacts/training/model.h5")
        Evaluate_model.Processing_Data()
        Evaluate_model.evaluation()
        # Evaluate_model.log_into_mlflow()
      except Exception as e:
        raise e
            

if __name__=='__main__':
    try :
          logger.info(f">>>>>>>Stage{stage_name} started <<<<<<<<<<<")
          obj=ModelEvaluationPipeleine()
          obj.main()
          logger.info(f">>>>>>>Stage{stage_name} completed <<<<<<<<<<<")
    except Exception as e :
         logger.exception(e)
         raise e
    

