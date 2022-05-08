import base64
import numpy as np
import cv2
import os
import json
import time

from os import listdir
from os.path import isdir

from tensorflow.keras.models import load_model
from PIL import Image
from numpy import asarray, savez_compressed, load, expand_dims
from mtcnn.mtcnn import MTCNN

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from random import choice
import uuid

import recognition_helper as recognition_helper

# load the model


def check_viewBlob(blob, detection, face_id):
    if detection == 'face':
        check = recognition_helper.face_detect(blob)
    else:
        check = recognition_helper.side_detect(blob)

    return check


# extract a single face from a given photograph
def extract_face(cvimage, required_size=(160, 160), save_image=False):
    # load image from file
    cvimage = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cvimage)
    # image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    if len(results) > 0:
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)

        if save_image is True:
            print('RETURN IMG')
            print(image)
            return np.asarray(image)
        else:
            face_array = asarray(image)
            return face_array

    return []


def load_faces(directory):
    faces = list()
    # enumerate files

    for filename in listdir(directory):
        # path
        path = directory + filename
        # load image from file
        image = Image.open(path)
        # convert to RGB, if needed
        image = image.convert('RGB')
        pix = np.array(image)
        # get face
        # store
        faces.append(pix)
    return faces


def clean_dir(user):
    path = '/dataset/val/'+str(user)
    files = os.listdir()

    for index, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.png'])))


def face_detect():
    faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # set Width
    cap.set(4, 480)  # set Height
    while True:
        ret, img = cap.read()
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
        cv2.imshow('video', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


def gather_data(id='Filip'):
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    # face_detector = cv2.CascadeClassifier(
    #    'Cascades/haarcascade_frontalface_default.xml')  # For each person, enter one numeric face id
    face_id = id
    print(
        "\n [INFO] Initializing face capture. Look the camera and wait ...")  # Initialize individual sampling face count
    count = 0
    retval = os.getcwd()
    print("Current working directory %s" % retval)
    os.mkdir(retval + "/dataset/train/" + str(id))
    os.mkdir(retval + "/dataset/val/" + str(id))
    while (True):
        ret, img = cam.read()
        # img = cv2.flip(img, -1)  # flip video image vertically
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = extract_face(img)
        count += 1
        # Save the captured image into the datasets folder
        print(r"dataset/train/" + str(id) + "/" + str(face_id) + str(count) + ".jpg")
        if count <= 30:
            print('writing')
            cv2.imwrite(r"dataset/train/" + str(id) + "/" + str(face_id) + str(count) + ".jpg", faces)
        # cv2.imwrite("dataset/" + str(id) + "/User." + str(face_id) + '.' +
        #            str(count) + ".jpg", faces)

        cv2.imshow('image', faces)
        print(count)
        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        print(r"dataset/val/" + str(id) + "/" + str(face_id) + str(count) + ".jpg")
        if count > 30:
            cv2.imwrite(r"dataset/val/" + str(id) + "/" + str(face_id) + str(count) + ".jpg", faces)

        if count >= 40:  # Take 30 face sample and stop video
            break  # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


def get_images_and_labels(path, detector):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceSamples, ids


def train_person():
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

    faces, ids = get_images_and_labels(path, detector)
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    recognizer.train(faces, np.array(ids))  # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')  # Print the number of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


def recognize():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX  # iniciate id counter
    id = 0  # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Filip', 'Filip', 'B', 'Z', 'W']  # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height# Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        ret, img = cam.read()
        # img = cv2.flip(img, -1)  # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            # If confidence is less them 100 ==> "0" : perfect match
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(
                img,
                str(id),
                (x + 5, y - 5),
                font,
                1,
                (255, 255, 255),
                2
            )
            cv2.putText(
                img,
                str(confidence),
                (x + 5, y + h - 5),
                font,
                1,
                (255, 255, 0),
                1
            )

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory, user=None):
    X, y = list(), list()
    photo_index = 0

    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        print(path)
        # skip any files that might be in the dir
        if not isdir(path):
            continue

        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        if subdir == user:
            photo_index = len(faces) - 1
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)

    print(X)
    print(y)
    if user is not None:
        return photo_index, asarray(X), asarray(y)
    else:
        return asarray(X), asarray(y)


def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def main_recognition_handler(user):
    loaded_model = load_model('facenet_keras.h5')
    trainX, trainy = load_dataset(str(os.getcwd()) + "/dataset/train/")
    photo_index, testX, testy = load_dataset(str(os.getcwd()) + "/dataset/val/", user)
    print((str(os.getcwd()) + "\\dataset\\val\\" + str(user) + "\\"))
    savez_compressed('compressed.npz', trainX, trainy, testX, testy)
    data = load('compressed.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(loaded_model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)

    # convert each face in the test set to an embedding
    newTestX = list()
    for face_pixels in testX:
        embedding = get_embedding(loaded_model, face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
    print(newTestX.shape)
    # save arrays to one file in compressed format
    savez_compressed('compressed-embeddings.npz', newTrainX, trainy, newTestX, testy)

    data_Embed = load('compressed-embeddings.npz')
    print(data_Embed)
    trainX_Embed, trainy_Embed, testX_Embed, testy_Embed = data_Embed['arr_0'], data_Embed['arr_1'], data_Embed[
        'arr_2'], data_Embed['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX_Embed.shape[0], testX_Embed.shape[0]))
    testX_faces = data['arr_2']
    print(trainX_Embed)
    in_encoder = Normalizer(norm='l2')
    trainX_Embed = in_encoder.transform(trainX_Embed)
    testX_Embed = in_encoder.transform(testX_Embed)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy_Embed)
    trainy_Embed = out_encoder.transform(trainy_Embed)
    testy_Embed = out_encoder.transform(testy_Embed)
    # fit model
    model = SVC(kernel='linear', C=1.0, gamma=0.01, probability=True)
    model.fit(trainX_Embed, trainy_Embed)
    # predict
    # yhat_train = model.predict(trainX)
    # yhat_test = model.predict(testX)
    # score
    # score_train = accuracy_score(trainy, yhat_train)
    # score_test = accuracy_score(testy, yhat_test)
    # summarize
    # print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
    # train_person()
    # recognize()

    selection = photo_index - 1
    random_face_pixels = testX_faces[selection]
    random_face_emb = testX_Embed[selection]
    random_face_class = testy_Embed[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])

    # prediction for the face
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)

    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    print('Expected: %s' % user)
    title = '%s (%.3f)' % (predict_names[0], class_probability)

    return title, predict_names[0]


def save_data(blob, face_id, directory):
    jpg_original = base64.b64decode(blob.split(',')[1])
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    print(jpg_as_np)
    img = cv2.imdecode(jpg_as_np, flags=1)
    faces = extract_face(img, required_size=(160, 160), save_image=True)
    print(faces)
    # Save the captured image into the datasets folder
    print("/dataset/" + directory + "/" + str(face_id) + "/" + str(face_id) + str(uuid.uuid1()) + ".png")
    print(faces)

    if not os.path.isdir(os.getcwd() + "/dataset/" + directory + "/" + str(face_id)):
        os.mkdir(os.getcwd() + "/dataset/" + directory + "/" + str(face_id))

    writeStatus = cv2.imwrite(os.getcwd() + "/dataset/" + directory + "/" + str(face_id) + "/" + str(face_id) + str(uuid.uuid1()) + ".png", faces)
    if writeStatus is True:
        print("image written")
    else:
        print("problem")  # or raise exception, handle problem, etc.
        print(writeStatus)
        print(os.path.dirname(os.path.realpath(__file__)))
        print(os.getcwd())
    return 'Saved'


def save_authentication(user):
    os.mkdir(os.getcwd() + "/auth/" + str(user))
    userDict = {
        "user": user,
        "created_at": time.now()
    }
    jsonString = json.dumps(userDict)
    jsonFile = open(os.getcwd() + "/auth/" + str(user) + "/data.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def delete_authentication(user):
    filePath = os.getcwd() + "/auth/" + str(user) + "/data.json"

    if os.path.exists(filePath):
        os.remove(filePath)
    else:
       return "Can not delete the file as it doesn't exists"

    return "Successfully deleted auth token"


def remove_user():
    print('Hello')