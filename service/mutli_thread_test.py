# coding=utf-8  
# @Time   : 2021/3/10 16:08
# @Auto   : zzf-jeff

import requests
import threading
import time

# KERAS_REST_API_URL = "http://localhost:8001/predict_v1"
KERAS_REST_API_URL = "http://localhost:8001/predict"
IMAGE_PATH = "test/test1.png"



class postrequests():
    def __init__(self):
        self.url = KERAS_REST_API_URL
        image = open(IMAGE_PATH, "rb").read()
        self.files = {"image": image}

    def post(self):
        try:
            for _ in range(50):
                r = requests.post(self.url, files=self.files)
                print(r.text)
        except Exception as e:
            print(e)


def login():
    login = postrequests()
    return login.post()


if __name__ == '__main__':
    try:
        i = 0
        # 开启线程数目
        tasks_number = 16
        ts = []
        time1 = time.time()
        while i < tasks_number:
            t = threading.Thread(target=login)
            ts.append(t)
            i += 1

        time1 = time.time()
        for t in ts:
            t.start()

        for t in ts:
            t.join()

        time2 = time.time()

        print(time2 - time1)
    except Exception as e:
        print(e)
