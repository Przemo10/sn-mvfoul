from decouple import config
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

TOTAL_DATASETS = ["train", "valid", "test", ]

def download_soccer_data(password, version):

    mySNdl = SNdl(LocalDirectory="./")

    mySNdl.downloadDataTask(
        task="mvfouls",
        split=TOTAL_DATASETS,
        password=password,
        version=version
    )


if __name__ == "__main__":
    password = config('DATASET_PASS')
    dataset_version = config('DATASET_VERSION')
    download_soccer_data(password, dataset_version)