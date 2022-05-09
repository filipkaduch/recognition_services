import base64
import io
import os
import sys

import time
import numpy as np
from PIL.Image import Image
from flask import Flask
from flask_cors import CORS, cross_origin

from flask import request
import json
import recognition_handler as recognition_service
# from pyngrok import ngrok

app = Flask(__name__)

cors = CORS(app)

# def init_webhooks(base_url):
#     # Update inbound traffic via APIs to use the public-facing ngrok URL
#     pass
#
# app.config.from_mapping(
#     BASE_URL="http://localhost:5000",
#     USE_NGROK=os.environ.get("USE_NGROK", "False") == "True" and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
# )
# when starting the server
# port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 5000
# ngrok.set_auth_token('23vx4PcbjWZ3JC8pQt4oIpSiNGD_7yboeKyaQkwrYh38oNy4s')
# Open a ngrok tunnel to the dev server
# public_url = ngrok.connect(port).public_url
# print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

# Update any base URLs or webhooks to use the public ngrok URL
# app.config["BASE_URL"] = public_url
# init_webhooks(public_url)

app.config['CORS_HEADERS'] = 'Content-Type'

# from app import routescd


def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return np.array(image)


@app.route('/check_view', methods=['POST'])
@cross_origin
def check_views():
    request_data = json.loads(request.data)
    return recognition_service.check_viewBlob(request_data['data']['image'], request_data['data']['detection']), 200


@app.route('/recognize_user', methods=['POST'])
@cross_origin
def recognize_users():
    request_data = json.loads(request.data)
    check = recognition_service.check_viewBlob(request_data['data']['image'], 'face', request_data['data']['user'])

    if check != 'Not detected':
        recognition_service.save_data(request_data['data']['image'], request_data['data']['user'], 'val'), 200
    else:
        return 'Not detected', 200
    recognized_text, predicted_name = recognition_service.main_recognition_handler(request_data['data']['user'])
    if predicted_name == request_data['data']['user']:
        recognition_service.save_authentication(request_data['data']['user'])
        # requests.post(url='http://147.175.105.115:8080/authy', data='Success')
    return recognized_text, 200


@app.route('/check_registration', methods=['GET'])
@cross_origin
def check_registrations():
    username = request.args.get('username')
    if os.path.isdir('dataset/train/' + username):
        if len(os.listdir('dataset/train/' + username)) > 10:
            return '1', 200
        else:
            return '0', 200
    else:
        return '0', 200


@app.route('/check_authentication', methods=['GET'])
@cross_origin
def check_authentications():
    username = request.args.get('username')
    if os.path.isdir('auth/' + username):
        f = open(os.getcwd() + "/auth/" + str(username) + "/data.json")
        data = json.load(f)
        f.close()
        recognition_service.delete_authentication(username)
        # print(time.time() - data['created_at'])
        if time.time() - data['created_at'] < 60:
            return '1', 200
    else:
        return '0', 200


@app.route('/register_user', methods=['POST'])
@cross_origin
def register_users():
    request_data = json.loads(request.data)
    fh = open("imageToSave.png", "wb")
    jpg_original = base64.decodebytes(str(request_data['data']['image'].split(',')[1]).encode())
    fh.write(jpg_original)
    fh.close()
    check = recognition_service.check_viewBlob(request_data['data']['image'], request_data['data']['detection'], request_data['data']['user'])

    if check != 'Not detected':
        return recognition_service.save_data(request_data['data']['image'], request_data['data']['user'], request_data['data']['directory']), 200
    else:
        return 'Not detected', 200
    # recognition_service.gather_data(request_data['data']['image'], request_data['data']['detection'])


@app.route('/delete_user', methods=['DELETE'])
@cross_origin
def delete_users():
    request_data = json.loads(request.data)
    return recognition_service.remove_user(request_data['user']), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')
