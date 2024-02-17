from data.exoplanet_data_frame import ExoplanetDataFrame
from model.model import UNet
from data.asteroid_data_frame import AsteroidDataFrame
from data.star_data_frame import StarDataFrame
#from train.train_data import trainData


asteroid_path = "C:\\Users\\SystemX\\Desktop\\projects\\UNet\\dataset\\asteroid_dataset\\dataset.csv"

star_path = "C:\\Users\\SystemX\\Desktop\\projects\\UNet\\dataset\\cleaned_star_data\\cleaned_star_data.csv"
exoplanet_path = "C:\\Users\\SystemX\\Desktop\\projects\\UNet\\dataset\\exo_dataset\\exoTrain.csv"

def main():
    asteroidData = AsteroidDataFrame(asteroid_path)
    asteroidData.train_data()

if __name__ == "__main__":
    main()