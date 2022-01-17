from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
import sqlalchemy as db
import json
import hashlib
import os
import src.services.recognition_service.recognition_handler as recognition_service
from flask_cors import CORS, cross_origin
import threading

from time import sleep


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# DB CONFIG
app.config['MYSQL_HOST'] = 'mysql80.r1.websupport.sk'
app.config['MYSQL_USER'] = '28ozvk5e'
app.config['MYSQL_PORT'] = 3314
app.config['MYSQL_PASSWORD'] = 'Ok12b\F9fm'
app.config['MYSQL_DB'] = '28ozvk5e'


engine = db.create_engine('mysql+pymysql://28ozvk5e:Ok12b\F9fm@mysql80.r1.websupport.sk:3314/28ozvk5e', echo=True)
mysql = MySQL(app)


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