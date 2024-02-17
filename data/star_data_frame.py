import pandas as pd
from sklearn.model_selection import train_test_split

class StarDataFrame:
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path) 
