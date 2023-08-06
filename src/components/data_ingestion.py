import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #ye class k variables ko initialize k sath sath value b assign karti hai(advance type of constructor)
#it is used to creat the class variables

@dataclass
class DataIngestionConfig:
    '''
    ye class basically humary data_ingustion class ko input provide karti hai.  
    '''
    train_data_path: str=os.path.join("artifacts", "train.csv")#is line ny train.csv file input li aur sath mien ye b define kar diya k output kahan save karni hai(artifact" folder mien)
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() #ab jasy he is class ka obj bna tu oper wali 3no input in ko mil jaien gi.

    def initiate_data_ingestion(self):
        #agar humara data kise database mien stored hai tu hum yahan read the data from database ka code likhien gy.
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/stud.csv') #ye hum data read kia (raw data)
            logging.info('Read the data as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train test split initiated')

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of the data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()

        


