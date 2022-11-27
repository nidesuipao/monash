import cv2
import os
import pickle
import numpy as np
import json
from skimage.feature import hog
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
import time

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse', 'ship','shiptruck']

def load_cifar10_batch(cifar10_dataset_folder_path):
    with open(cifar10_dataset_folder_path, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    # features and labels
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

CIFAR_train = './data/cifar10/data_batch_2'
CIFAR_test = './data/cifar10/test_batch'
train_data, train_label = load_cifar10_batch(CIFAR_train)
test_data, test_label = load_cifar10_batch(CIFAR_test)
detector = cv2.SIFT_create()

def create_dictionary(clust_class, feature_type):
    z = 4
    bag_of_word = []
    if feature_type == 'hog':
        for img in train_data:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            feature = hog(gray, pixels_per_cell=(4, 4), cells_per_block=(z, z), feature_vector=True).reshape(-1, z * z * 9)
            bag_of_word.append(feature)
        bag_of_words = np.array(bag_of_word).reshape(-1, z * z * 9)
    else:
        for img in train_data:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            if len(keypoints) != 0:
                for des in descriptors:
                    bag_of_word.append(des)
        bag_of_words = np.array(bag_of_word).reshape(-1, 128)
    print("Done BOW..")
    kmeans_vocab = MiniBatchKMeans(n_clusters=clust_class, init='k-means++', random_state=np.random.RandomState(0)).fit(
        bag_of_words)
    print("Done Kmeans..")
    centroids = kmeans_vocab.cluster_centers_
    return centroids

def compute_histogram(img, centers, clust_class, feature_type):
    histogram = np.zeros((clust_class))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if feature_type == 'hog':
        feature = hog(gray, pixels_per_cell=(4, 4), cells_per_block=(4, 4), feature_vector=True).reshape(-1, 144)
        all_distances = cdist(feature, centers, 'cosine')
        for feat_index in range(all_distances.shape[0]):
            min_distance_index = np.argsort(all_distances[feat_index, :])[0]
            histogram[min_distance_index] += 1
    else:
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        if len(keypoints) != 0:
            all_distances = cdist(descriptors, centers, 'cosine')
            for feat_index in range(all_distances.shape[0]):
                min_distance_index = np.argsort(all_distances[feat_index, :])[0]
                histogram[min_distance_index] += 1
    return histogram

def get_bags_of_words(data, clust_class, feature_type):
    centers = np.load('centers.npy')
    indecies_hist = np.zeros((len(data), clust_class))
    image_index = -1
    for img in data:
        image_index += 1
        indecies_hist[image_index] = compute_histogram(img, centers, clust_class,feature_type)
    return indecies_hist


def match_histogram(histogram1, histogram2):
    return np.linalg.norm((histogram1 - histogram2), ord=2)

def predict_knn(train_hist, train_label, test_hist, k_neighbour):
    n_test = test_hist.shape[0]
    distances = np.zeros((len(test_hist), len(train_hist)))
    test_labels = []
    for i in range(len(test_hist)):
        for j in range(len(train_hist)):
            distances[i][j] = match_histogram(test_hist[i], train_hist[j])

    for i in range(n_test):
        indexs = np.argsort(distances[i])[0:k_neighbour]
        labels = [train_label[index] for index in indexs]
        test_labels.append(np.argmax(np.bincount(labels)))
    test_labels = np.array(test_labels)
    return test_labels

def class_wise(train_hist, train_label, test_hist, k_neighbour,clust_class):
    print("clust_class: ", str(clust_class), ", k_neighbour:", str(k_neighbour))
    test_labels = np.array(test_label).reshape(-1)
    class_test_data = [[] for i in range(10)]
    class_test_acc = [0 for i in range(10)]
    for i in range(len(test_hist)):
        class_test_data[test_labels[i]].append(test_hist[i])
    for i in range(10):
        class_test_data[i] = np.array(class_test_data[i])
        pre_labels = predict_knn(train_hist, train_label, class_test_data[i], k_neighbour)
        class_test_acc[i] = np.sum(pre_labels == i)/pre_labels.shape[0]
        print('the accuracy of ', classes[i], 'is :', str(class_test_acc[i]))
    acc = 0
    for i in range(10):
        acc += class_test_acc[i] * len(class_test_data[i])
    print("overall accuracy:", str(acc/len(test_hist)))




def acc_compute(clust_class, k_neighbour, feature_type = 'hog', is_test = True):
    now = time.time()
    clust_changed = False
    if is_test:
        print("Start to create dictionary ..")
    if not os.path.isfile('centers.npy'):
        if is_test:
            print("centers file not exist..")
        centers = create_dictionary(clust_class, feature_type)
        np.save('centers.npy', centers)
    else:
        centers = np.load('centers.npy')
        if centers.shape[0] != clust_class:
            if is_test:
                print("clust_class changed, need to recreate centers..")
            clust_changed = True
            centers = create_dictionary(clust_class, feature_type)
            np.save('centers.npy', centers)
    if is_test:
        print("create the dictionary successfully, cost time:", str(time.time() - now), 'second')
        now = time.time()
    if not os.path.isfile('train_hist.npy') or clust_changed:
        train_hist = get_bags_of_words(train_data, clust_class, feature_type)
        np.save('train_hist.npy', train_hist)
    else:
        train_hist = np.load('train_hist.npy')

    if not os.path.isfile('test_hist.npy') or clust_changed:
        test_hist = get_bags_of_words(test_data, clust_class, feature_type)
        np.save('test_hist.npy', test_hist)
    else:
        test_hist = np.load('test_hist.npy')
    if is_test:
        print("create the histogram successfully, cost time:", str(time.time() - now), 'second')
        now = time.time()

    if not is_test:
        test_hist = test_hist[0:100]
        pre_labels = predict_knn(train_hist, train_label, test_hist, k_neighbour)
        test_labels = np.array(test_label).reshape(-1)
        count = 0
        for  i in range(pre_labels.shape[0]):
            count += pre_labels[i] == test_labels[i]
        return count / pre_labels.shape[0]
    if is_test:
        class_wise(train_hist, train_label, test_hist, k_neighbour, clust_class)
        print("predict finished, cost time:", str(time.time() - now), 'second')


# accuracys = {}
# for n in range(10,200,10):
#     for k in range(10,200,10):
#         acc  = acc_compute(n, k,'hog', False)
#         accuracys[str(n) + ',' + str(k)] = acc
#
# with open("accuracys.json", "w+") as json_file:
#     accuracys = json.dumps(accuracys)
#     json_file.write(accuracys)

with open("accuracys.json", "r") as json_file:
    data = json_file.read()
    accuracys = json.loads(data)
    keys = list(accuracys.keys())
    values = list(accuracys.values())
    (n,k) = keys[values.index(max(values))].split(',')
    acc_compute(int(n), int(k), 'hog', True)




