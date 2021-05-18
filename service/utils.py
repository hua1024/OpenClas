# coding=utf-8  
# @Time   : 2020/9/24 10:07
# @Auto   : zzf-jeff
import base64
from io import BytesIO
import cv2
import numpy as np

import requests as req
from PIL import Image
from io import BytesIO


def url2img(img_url):
    '''
    url图片转cv2
    '''
    response = req.get(img_url)
    image = Image.open(BytesIO(response.content))
    image = cv2.cvtColor(np.asanyarray(image), cv2.COLOR_RGB2BGR)
    return image


def img2base64(image, img_format='PNG'):
    '''
    cv2格式图片转base64
    '''
    try:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_buffer = BytesIO()
        image.save(img_buffer, format=img_format)
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)
        base64_str = str(base64_str, "utf-8")  # bytes -> base64 string
        return base64_str
    except Exception as e:
        return None


#


def base64_img(str_base64):
    '''
    base64图片解码成cv2格式
    '''
    img_b64decode = base64.b64decode(str_base64)  # base64解码

    img_array = np.fromstring(img_b64decode, np.uint8)  # 转换np序列
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)  # 转换Opencv格式
    return img


