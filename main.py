from model.model import UNet
from data.asteroid_data_frame import AsteroidDataFrame


path = "C:\\Users\\SystemX\\Desktop\\projects\\UNet\\dataset\\asteroid_dataset\\dataset.csv"

def main():
    model = UNet()
    data = AsteroidDataFrame(path)
    print(data.df.info())


if __name__ == "__main__":
    main()