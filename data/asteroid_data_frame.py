import pandas as pd
import pandas as pd
import numpy as np
from feature_engine.encoding import RareLabelEncoder
from sklearn.model_selection import train_test_split

#y = df[main_label].values.reshape(-1,)  # Reshape the target variable as a 1D array
#X = df.drop([main_label], axis=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


class AsteroidDataFrame:
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path)
        self.main_label = 'is_hazardeous'

    def data_frame_optim(self):
        item0 = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        item1 = self.df.shape[0]

        self.df = self.df[self.df['pha'].isin(['Y', 'N'])]
        self.df['is_hazardeous'] = (self.df['pha'] == 'Y').astype(int)

        self.df['orbit_id'] = self.df['orbit_id'].apply(lambda x: x.replace('JPL', '').strip())

        selected_cols = ['is_hazardeous', 'neo', 'H', 'diameter', 'albedo', 'orbit_id', 'e', 'q', 'i', 'om', 'w',
       'ad', 'n', 'per', 'per_y', 'moid_ld', 'class', 'rms']
    
        self.df = self.df[selected_cols]

    def data_frame_transformation(self):

        def group_bin(x, delta=2):
            try:
                return str(delta*round(x/delta))
            except:
                return 'None'
            
        for col in ['H', 'i', 'om', 'w']:
            self.df[col] = self.df[col].apply(group_bin)
            print(f"Binned column {col}")

        def log10_bin(x):
            try:
                return str(round(1/5*round(5*np.log10(x+1)),1))
            except:
                return 'None'
            
        for col in ['diameter', 'albedo', 'e', 'q', 'ad', 'n', 'per', 'per_y', 'moid_ld', 'rms']:
            self.df[f'log10_{col}'] = self.df[col].apply(log10_bin)
            self.df = self.df.drop([col], axis=1)
            print(f"Log10-transformed column {col}")

        for col in self.df.columns:
            if col != self.main_label:
                self.df[col] = self.df[col].fillna('None').astype(str)
                encoder = RareLabelEncoder(n_categories=1, max_n_categories=150, replace_with='Other', tol=100/self.df.shape[0])
                self.df[col] = encoder.fit_transform(self.df[[col]])
                print(f"LabelEncoded column {col}")
    
    def train_data(self):
        self.data_frame_optim()
        self.data_frame_transformation()

        y = self.df[self.main_label].values.reshape(-1,)
        X = self.df.drop([self.main_label], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        print("Shape of X_train:", X_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of y_test:", y_test.shape)