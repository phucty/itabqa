import gdown

url = "https://drive.google.com/drive/u/1/folders/14IAqJb9ObVDE5oOJouhkqgd_mn11PkYY"
gdown.download(url=url, resume=True, quiet=False)
