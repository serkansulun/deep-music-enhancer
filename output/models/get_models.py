""" Downloads pretrained models from Google Drive
https://stackoverflow.com/a/39225272 
It might take a while since each model is around 600 MB"""

import requests
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--model", type=str, required=True,
                    help="""Model to download. Options: 
                    unet (Baseline U-Net)
                    unetda (U-Net with data augmentation)
                    unetbn (U-Net with batch normalization)
                    unetdo (U-Net with dropout)
                    resnet (Baseline ResNet)
                    resnetda (ResNet with data augmentation)
                    resnetbn (ResNet with batch normalization)
                    resnetdo (ResNet with dropout)
                    """)
args = parser.parse_args()

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":

    ids = {
        "unet": "1cg2BDS2WqwIvicihDJFxak7FiffQMGdu",
        "unetda": "1iKyGo_28NvRzOY6JKNPdf9FLnOrGH_GN",
        "unetbn": "1Altk_WsfpdDHWDPg_bVuqwC7rK0kmmle",
        "unetdo": "1bC6wPdIDS5kMwuWOpckqsNOPHiZ2orYe",
        "resnet": "1HcOTwYjR23ZLs0VVMymiUYbDb6FXhgz9",
        "resnetda": "18UnGjoHE16v76w7qMrTEiTgsUelDG1fY",
        "resnetbn": "1JjLQPqH0qW_l-P1jpNOTBbvxtKDJknDS",
        "resnetdo": "1ycpnQj98bi2eEsLef0ZIRBmPI5UjMJ5Y",
    }

    if args.model in ids.keys():
        file_id = ids[args.model]
        destination = args.model + ".pt"
        download_file_from_google_drive(file_id, destination)
    else:
        raise ValueError("""Invalid model. Options: 
                    unet (Baseline U-Net)
                    unetda (U-Net with data augmentation)
                    unetbn (U-Net with batch normalization)
                    unetdo (U-Net with dropout)
                    resnet (Baseline ResNet)
                    resnetda (ResNet with data augmentation)
                    resnetbn (ResNet with batch normalization)
                    resnetdo (ResNet with dropout)
                    """)