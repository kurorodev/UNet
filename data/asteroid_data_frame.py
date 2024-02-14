import pandas as pd
from sklearn.model_selection import train_test_split

#y = df[main_label].values.reshape(-1,)  # Reshape the target variable as a 1D array
#X = df.drop([main_label], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


class AsteroidDataFrame:
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path)