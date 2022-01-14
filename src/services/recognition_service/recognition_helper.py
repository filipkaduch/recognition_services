import base64
import io

import numpy as np
import cv2
import os
from PIL import Image
from flask import make_response


def face_detect(string):
    faceCascade = cv2.CascadeClassifier('src/constants/recognition/cascades/haarcascade_frontalface_default.xml')
    jpg_original = base64.b64decode(string.split(',')[1])
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    retval, buffer = cv2.imencode('.jpg', img)
    response = make_response(buffer.tobytes())
    if len(faces) > 0:
        return response
    else:
        return 'Not detected'


def side_detect(string):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    jpg_original = base64.b64decode(string.split(',')[1])
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    retval, buffer = cv2.imencode('.jpg', img)
    response = make_response(buffer.tobytes())
    if len(faces) > 0:
        return response
    else:
        return 'Not detected'

