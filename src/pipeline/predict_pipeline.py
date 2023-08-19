import sys
import pandas as pd
import os

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        #here we will perform prediction (ye function sb say end likha hai)
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class Customdata:
    '''
    this class is reponsible in mapping all the input we are giving in the html to particular backend values

    '''
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education, #agar hum datatype na bi dien tu its fine
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int): #hum ny sath mien datatype b mention kar diya hai k in variable ki values ye must honi chahy(str,int)

        #in variable k name same honay chahy us names k sath jo hum ny html mien define kiay huay hain e.g <select class="form-control" name="gender" placeholder="Enter you Gender" required>
        
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score 


    def get_data_as_data_frame(self):

        '''
        this function will return all the input in the form of  dataframe because we train our model in the form of dataframe.
        inshort jo b input hum apni web page say dien gy wo datafrme mien convert ho jaien gi.

        '''


        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)