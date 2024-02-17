import sys
sys.path.append("C:/Users/SystemX/Desktop/projects/UNet")
from sklearn.model_selection import train_test_split
import pandas as pd
from data.asteroid_data_frame import AsteroidDataFrame

asteroid_path = "C:\\Users\\SystemX\\Desktop\\projects\\UNet\\dataset\\asteroid_dataset\\dataset.csv"

def trainData():
    asteroidData = AsteroidDataFrame(asteroid_path)
    asteroidData.data_frame_optim()
    asteroidData.data_frame_transformation()

    y = asteroidData.df[asteroidData.main_label].values.reshape(-1,)
    X = asteroidData.df.drop([asteroidData.main_label], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
