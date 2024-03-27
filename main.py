from classification.pipeline.stage02_Preparation_baseModel import PreparationBaseModelPipeleine
from classification.pipeline.Stage01_Data_Ingestion import DataIngestionTrainingPipeleine
from classification.pipeline.stage03_Training import TrainignModelPipeleine
from src.classification.pipeline.Stage04_Model_Evaluation import ModelEvaluationPipeleine
from classification import logger

stage_name="Data ingestion "
try :
      logger.info(f">>>>>>>Stage{stage_name} started <<<<<<<<<<<")
      obj=DataIngestionTrainingPipeleine()
      obj.main()
      logger.info(f">>>>>>>Stage{stage_name} completed <<<<<<<<<<<")
except Exception as e :
      logger.exception(e)
      raise e


stage_name="Model preparation "

try :
      logger.info(f">>>>>>>Stage{stage_name} started <<<<<<<<<<<")
      obj=PreparationBaseModelPipeleine()
      obj.main()
      logger.info(f">>>>>>>Stage{stage_name} completed <<<<<<<<<<<")
except Exception as e :
      logger.exception(e)
      raise e
stage_name="Training  Model"
try :
      logger.info(f">>>>>>>Stage{stage_name} started <<<<<<<<<<<")
      obj=TrainignModelPipeleine()
      obj.main()
      logger.info(f">>>>>>>Stage{stage_name} completed <<<<<<<<<<<")
except Exception as e :
      logger.exception(e)
      raise e

stage_name="Evaluate Model"

try :
          logger.info(f">>>>>>>Stage{stage_name} started <<<<<<<<<<<")

          obj=ModelEvaluationPipeleine()
          obj.main()
          logger.info(f">>>>>>>Stage{stage_name} completed <<<<<<<<<<<")
except Exception as e :
         logger.exception(e)
         raise e

