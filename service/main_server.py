# coding=utf-8  
# @Time   : 2021/3/6 14:33
# @Auto   : zzf-jeff

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('../')
from service.config import opt
from tools.deploy.trt_inference import TRTModel

from torchclas.models import build_backbone
from torchclas.utils.io_func import config_load

from PIL import Image
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import cv2
import sys
import io
import torch
from torchvision import transforms
import torch.nn.functional as F
from threading import Thread


# def base64_encode_image(a):
#     # base64 encode the input NumPy array
#     return base64.b64encode(a).decode("utf-8")


def base64_encode_image(im):
    """
    way4: PIL.Image-->二进制数据-->base64(str)
    :param im_path:
    :return:
    """
    imByteArr = io.BytesIO()
    im.save(imByteArr, format="JPEG")
    imByteArr = imByteArr.getvalue()
    im_base64 = base64.b64encode(imByteArr)
    im_base64 = str(im_base64, encoding="utf-8")
    return im_base64


# def base64_decode_image(a):
#     # a = bytes(a, encoding="utf-8")
#     # convert the string to a NumPy array using the supplied data
#     # type and target shape
#     a = np.fromstring(base64.b64decode(a), dtype=np.uint8)
#     a = cv2.imdecode(a, -1)  # return the decoded image
#     return a

def base64_decode_image(a):
    byte_data = base64.b64decode(a)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data)
    return img


def init_model(cfg, weight, engine, mode='torch'):
    if mode == 'torch':
        model = build_backbone(cfg['BACKBONES'])
        model.load_state_dict(torch.load(weight, map_location=device)['state_dict'])
        model = model.to(device)
        model.eval()
        return model
    else:
        trt = TRTModel(engine)
        return trt


def prepare_image(image):
    normalize_imgnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        normalize_imgnet
    ])
    image = trans(image)
    image = image.unsqueeze(0)
    return image


def classify_process(model, image, mode='torch'):
    if mode == 'torch':
        with torch.no_grad():
            image = image.to(device)
            output = model(image)
    else:
        output = model.run(image)
        output = torch.Tensor(output)

    result = {}
    prob = F.softmax(output, dim=1)
    value, predicted = torch.max(output.data, 1)
    pred_class = ['dog', 'cat'][predicted.item()]
    pred_score = prob[0][predicted.item()].item()
    result['class'] = pred_class
    result['score'] = pred_score
    return result


def classify_process_v1():
    while True:

        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(opt.queue, 0, opt.batch_size - 1)
        imageIDs = []
        batch = None
        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"])
            image = prepare_image(image)
            # check to see if the batch list is None
            if batch is None:
                batch = image
            # otherwise, stack the data
            else:
                batch = torch.from_numpy(np.vstack([batch, image]))
            # update the list of image IDs
            imageIDs.append(q["id"])
        # check to see if we need to process the batch
        if len(imageIDs) > 0:
            print("* Batch size: {}".format(batch.shape))
            if opt.mode == 'torch':
                with torch.no_grad():
                    batch = batch.to(device)
                    preds = model(batch)
            else:
                preds = model.run(batch)
                preds = torch.Tensor(preds)

            for (imageID, pred) in zip(imageIDs, preds):
                output = []
                result = {}
                prob = F.softmax(pred, dim=0)
                value, predicted = torch.max(pred.data, 0)
                pred_class = ['dog', 'cat'][predicted.item()]
                pred_score = prob[predicted.item()].item()

                result['class'] = pred_class
                result['score'] = pred_score
                output.append(result)
                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(output))
            # remove the set of images from our queue
            db.ltrim(opt.queue, len(imageIDs), -1)
        # sleep for a small amount
        time.sleep(opt.client_time)


def pil2cv2(image):
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return img


app = flask.Flask(__name__)
pool = redis.ConnectionPool(host='localhost', port=6379, max_connections=50)
db = redis.StrictRedis(connection_pool=pool, decode_responses=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cfg = config_load(opt.config)
model = init_model(cfg, opt.weight, opt.engine, mode=opt.mode)


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format and prepare it for
            # classification
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image)
            result = classify_process(model, image, mode=opt.mode)
            data["success"] = True
            data['predictions'] = result

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


@app.route("/predict_v1", methods=["POST"])
def predict_v1():
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format and prepare it for
            # classification
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            # image = prepare_image_v1(image)
            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())
            d = {"id": k, "image": base64_encode_image(image)}
            db.rpush(opt.queue, json.dumps(d))
            # keep looping until our model server returns the output
            # predictions
            while True:
                output = db.get(k)
                if output is not None:
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
            time.sleep(opt.client_time)
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    # load the function used to classify input images in a *separate*
    # thread than the one used for main classification
    # print("* Starting model service...")
    # t = Thread(target=classify_process_v1, args=())
    # t.daemon = True
    # t.start()

    # start the web server
    print("* Starting web service...")
    app.run(host='localhost', port=8001, debug=False, threaded=False)
