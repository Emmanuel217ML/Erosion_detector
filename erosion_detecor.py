import os 
import cv2
import numpy as np
from PIL import Image
import pickle
from img2vec_pytorch import Img2Vec
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


img2vec = Img2Vec()
 
data_dir = 'dataset'
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')

data = {}

for j, dir_ in enumerate([train_dir, validation_dir],):
    features = []
    lablels = []

    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path = os.path.join(dir_, category, img_path)
            img = Image.open(img_path)

            img_features = img2vec.get_vec(img)

            features.append(img_features)
            lablels.append(category)
    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = lablels


model = RandomForestClassifier()
model.fit(data['training_data'], data['training_labels'])

y_pred = model.predict(data['validation_data'])

score = accuracy_score ( y_pred, data['validation_labels'])

print(f'Accuracy: {score * 100:.2f}% were correctly classified.')
with open('erosion_detector_img2vec_model.p', 'wb') as f:
    pickle.dump(model, f)


    