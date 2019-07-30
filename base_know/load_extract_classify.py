# 里面可以学习到一些分类标签的提取和特征提取以及简单训练和判别
# coding=utf-8



import os

import sys
import argparse
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
# from star_detector import StarFeatureDetector
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

import pickle  # python 3


def load_training_data(input_folder):
    training_data = []
    if not os.path.isdir(input_folder):
        raise IOError("The folder " + input_folder + " doesn't exist")

    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if x.endswith('.png')):
            filepath = os.path.join(root, filename)
            print(filepath)
            object_class = filepath.split('/')[-2]
            print("object_class", object_class)
            training_data.append({'object_class': object_class, 'image_path': filepath})

    return training_data

class StarFeatureDetector(object):
    def __init__(self):
        self.detector = cv2.xfeatures2d.StarDetector_create()

    def detect(self, img):
        return self.detector.detect(img)


class FeatureBuilder(object):
    def extract_features(self, img):
        keypoints = StarFeatureDetector().detect(img)
        keypoints, feature_vectors = compute_sift_features(img, keypoints)
        return feature_vectors

    def get_codewords(self, input_map, scaling_size, max_samples=12):
        keypoints_all = []
        count = 0
        cur_class = ''
        for item in input_map:
            if count >= max_samples:
                if cur_class != item['object_class']:
                    count = 0
                else:
                    continue
            count += 1
            if count == max_samples:
                print("Built centroids for", item['object_class'])

            cur_class = item['object_class']
            img = cv2.imread(item['image_path'])
            img = resize_image(img, scaling_size)
            num_dims = 128
            feature_vectors = self.extract_features(img)
            keypoints_all.extend(feature_vectors)

        kmeans, centroids = BagOfWords().cluster(keypoints_all)
        return kmeans, centroids


class BagOfWords(object):
    def __init__(self, num_clusters=32):
        self.num_dims = 128
        self.num_clusters = num_clusters
        self.num_retries = 10

    def cluster(self, datapoints):
        kmeans = KMeans(self.num_clusters,
                        n_init=max(self.num_retries, 1),
                        max_iter=10, tol=1.0)
        res = kmeans.fit(datapoints)
        centroids = res.cluster_centers_
        return kmeans, centroids

    def normalize(self, input_data):
        sum_input = np.sum(input_data)

        if sum_input > 0:
            return input_data / sum_input
        else:
            return input_data

    def construct_feature(self, img, kmeans, centroids):
        keypoints = StarFeatureDetector().detect(img)
        keypoints, feature_vectors = compute_sift_features(img, keypoints)
        labels = kmeans.predict(feature_vectors)
        feature_vector = np.zeros(self.num_clusters)

        for i, item in enumerate(feature_vectors):
            feature_vector[labels[i]] += 1

        feature_vector_img = np.reshape(feature_vector, ((1, feature_vector.shape[0])))
        return self.normalize(feature_vector_img)


# Extract features from the input images and
# map them to the corresponding object classes
def get_feature_map(input_map, kmeans, centroids, scaling_size):
    feature_map = []
    for item in input_map:
        temp_dict = {}
        temp_dict['object_class'] = item['object_class']

        print("Extracting features for", item['image_path'])
        img = cv2.imread(item['image_path'])
        img = resize_image(img, scaling_size)

        temp_dict['feature_vector'] = BagOfWords().construct_feature(img, kmeans, centroids)
        if temp_dict['feature_vector'] is not None:
            feature_map.append(temp_dict)
    return feature_map


# Extract SIFT features
def compute_sift_features(img, keypoints):
    if img is None:
        raise TypeError('Invalid input image')

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.xfeatures2d.SIFT_create().compute(img_gray, keypoints)
    return keypoints, descriptors


# Resize the shorter dimension to 'new_size'
# while maintaining the aspect ratio
def resize_image(input_img, new_size):
    h, w = input_img.shape[:2]
    scaling_factor = new_size / float(h)

    if w < h:
        scaling_factor = new_size / float(w)

    new_shape = (int(w * scaling_factor), int(h * scaling_factor))
    return cv2.resize(input_img, new_shape)


def build_features_main():
    data_folder = '/home/osk/图片/ocr/good'
    scaling_size = 200
    codebook_file = 'codebook.pkl'
    feature_map_file = 'feature_map.pkl'
    # Load the training data
    training_data = load_training_data(data_folder)

    # Build the visual codebook
    print("====== Building visual codebook ======")
    kmeans, centroids = FeatureBuilder().get_codewords(training_data, scaling_size)
    if codebook_file:
        with open(codebook_file, 'wb') as f:
            pickle.dump((kmeans, centroids), f)

    # Extract features from input images
    print("\n====== Building the feature map ======")
    feature_map = get_feature_map(training_data, kmeans, centroids, scaling_size)
    if feature_map_file:
        with open(feature_map_file, 'wb') as f:
            pickle.dump(feature_map, f)


# --feature-map-file feature_map.pkl --model- file erf.pkl
# ----------------------------------------------------------------------------------------------------------
class ERFTrainer(object):
    def __init__(self, X, label_words):
        self.le = preprocessing.LabelEncoder()
        self.clf = ExtraTreesClassifier(n_estimators=100,
                                        max_depth=16, random_state=0)

        y = self.encode_labels(label_words)
        self.clf.fit(np.asarray(X), y)

    def encode_labels(self, label_words):
        self.le.fit(label_words)
        return np.array(self.le.transform(label_words), dtype=np.float32)

    def classify(self, X):
        label_nums = self.clf.predict(np.asarray(X))
        label_words = self.le.inverse_transform([int(x) for x in label_nums])
        return label_words


# ------------------------------------------------------------------------------------------

class ImageTagExtractor(object):
    def __init__(self, model_file, codebook_file):
        with open(model_file, 'rb') as f:
            self.erf = pickle.load(f)

        with open(codebook_file, 'rb') as f:
            self.kmeans, self.centroids = pickle.load(f)

    def predict(self, img, scaling_size):
        img = resize_image(img, scaling_size)
        feature_vector = BagOfWords().construct_feature(
            img, self.kmeans, self.centroids)
        image_tag = self.erf.classify(feature_vector)[0]
        return image_tag


def train_Recognizer_main():
    feature_map_file = 'feature_map.pkl'
    model_file = 'erf.pkl'
    # Load the feature map
    with open(feature_map_file, 'rb') as f:
        feature_map = pickle.load(f)
    # Extract feature vectors and the labels
    label_words = [x['object_class'] for x in feature_map]
    dim_size = feature_map[0]['feature_vector'].shape[1]
    X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map]

    # Train the Extremely Random Forests classifier
    erf = ERFTrainer(X, label_words)
    if model_file:
        with open(model_file, 'wb') as f:
            pickle.dump(erf, f)
    # --------------------------------------------------------------------
    # args = build_arg_parser().parse_args()
    model_file = 'erf.pkl'
    codebook_file = 'codebook.pkl'
    import os
    # rootdir = r"F:\airplanes"
    rootdir = '/home/osk/图片/ocr/bad'
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            try:
                print(path)
                input_image = cv2.imread(path)
                scaling_size = 200
                print("\nOutput:", ImageTagExtractor(model_file, codebook_file) \
                      .predict(input_image, scaling_size))
            except:
                continue
    # -----------------------------------------------------------------------


build_features_main()
train_Recognizer_main()

#
# import cv2
# import numpy as np
# from models import Model
# img = cv2.imread('/home/osk/图片/ocr/bad/bad_001.png',3)
# kernel = np.ones((7,7),np.uint8)
# dilation = cv2.dilate(img,kernel,iterations = 1)
#
# cv2.imshow('img',img)
# cv2.imshow('dil',dilation)
#
# cv2.waitKey(0)
