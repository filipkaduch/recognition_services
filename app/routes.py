from app import app
from app import cors
from flask import request, jsonify
import json
import hashlib
import os
import src.services.recognition_service.recognition_handler as recognition_service
from flask_cors import CORS, cross_origin
import threading

from time import sleep


@app.route('/check_view', methods=['POST'])
@cross_origin()
def check_view():
    request_data = json.loads(request.data)
    return recognition_service.check_view(request_data['data']['image'], request_data['data']['detection']), 200


@app.route('/recognize_user', methods=['POST'])
@cross_origin()
def check_view():
    request_data = json.loads(request.data)
    return recognition_service.main_recognition_handler(request_data['data']['image'], request_data['data']['detection']), 200


@app.route('/register_user', methods=['POST'])
@cross_origin()
def check_view():
    request_data = json.loads(request.data)
    print(request_data)
    return recognition_service.gather_data(request_data['data']['image'], request_data['data']['detection']), 200


@app.route('/delete_user', methods=['DELETE'])
@cross_origin()
def check_view():
    request_data = json.loads(request.data)
    return recognition_service.remove_user(request_data['userId']), 200


if __name__ == '__main__':
    app.run()