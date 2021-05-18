import requests
import time

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:8001/predict"
IMAGE_PATH = "test/test1.png"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()

payload = {"image": image}

# submit the request

start_time = time.time()

for _ in range(100):

    r = requests.post(KERAS_REST_API_URL, files=payload).json()
    # ensure the request was sucessful
    if r["success"]:
        print(r['predictions'])
    else:
        print("Request failed")

end_time = time.time()

print(end_time-start_time)