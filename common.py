import os

import pandas as pd
import requests
import tensorflow as tf
from flair_cosine_similarity import flair_semantic
from elmo_cosine_similarity import elmo_semantic

class similarity_test:
    def download_sick_dataset(self,url):
        response = requests.get(url).text

        lines = response.split("\n")[1:]
        lines = [l.split("\t") for l in lines if len(l) > 0]
        lines = [l for l in lines if len(l) == 5]

        df = pd.DataFrame(lines, columns=["idx", "sent_1", "sent_2", "sim", "label"])
        df['sim'] = pd.to_numeric(df['sim'])
        return df

    def normalize(self,df, feature_names):
        result = df.copy()
        for feature_name in feature_names:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    def download_and_load_sick_dataset(self):
        sick_train = self.download_sick_dataset(
            "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt")
        sick_dev = self.download_sick_dataset(
            "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt")
        sick_test = self.download_sick_dataset(
            "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_test_annotated.txt")
        sick_all = sick_train.append(sick_test).append(sick_dev)

        return sick_all

    def 
