from classification.components.Data_ingestion import *
from classification import logger

stage_name="Data ingestion stage"

class DataIngestionTrainingPipeleine:
    def __init__(self) :
        pass
    def main(self):
              
      
          config = configurationManager()
          data_ingestion_config = config.data_ingestion_configuration()
          data_ingestion = DataIngestion(config=data_ingestion_config)
          data_ingestion.download_file()
          data_ingestion.extract_zip_file()
      

if __name__=='__main__':
    try :
          logger.info(f">>>>>>>Stage{stage_name} started <<<<<<<<<<<")
          obj=DataIngestionTrainingPipeleine()
          obj.main()
          logger.info(f">>>>>>>Stage{stage_name} completed <<<<<<<<<<<")
    except Exception as e :
         logger.exception(e)
         raise e
    
    




        

