import csv
import os
import random
from os import listdir
import cv2
import numpy as np
from colorama import Fore
from keras.preprocessing.image import load_img, img_to_array
from natsort import natsorted
from tensorflow.keras.utils import Sequence, to_categorical

global ext_img


class DataGenerator(Sequence):

    def __init__(self, data, labels, batch_size=32, dim=(20, 120, 120, 3), n_channels=1,
                 n_classes=51, shuffle=True):
        self.data = data
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = np.arange(len(data))
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        global ext_img
        X_data = []
        y_data = []
        mydata = self.data
        # interate through  each sequence
        for i, id_batch in enumerate(list_IDs_temp):
            seq_frames = mydata.iloc[0]
            y = mydata.iloc[1]
            temp_data_list = []
            for img in seq_frames:
                image = cv2.imread(img)
                print(img)
                print(image.shape)
                ext_img = cv2.resize(image, self.dim)
                temp_data_list.append(ext_img)
            X_data.append(temp_data_list)
            y_data.append(y)
        X = np.array(X_data)  # Converting list to array
        y = np.array(y_data)
        return X, to_categorical(y, num_classes=self.n_classes)


def PickingImages(camera_folder_name):
    print(Fore.LIGHTYELLOW_EX + " ========= READ FILES PROCESS IS STARTING ========")
    # pick names and labels
    trainyNames = []
    labels = []
    count = 0
    for entry2 in os.listdir(camera_folder_name):
        action_sec_folder = os.path.join(camera_folder_name, entry2)
        if os.path.isdir(action_sec_folder):
            splitNameAction = entry2.split("_")
            list_of_frames = natsorted(os.listdir(action_sec_folder))
            number_files = len(list_of_frames)
            size = number_files // 20
            if number_files < 20:
                count = count + 1
            else:
                for i in range(0, 20):  # put frames into segments
                    segment = list_of_frames[i * size:i * size + size]
                    randomIndex = random.randint(0, size - 1)
                    trainyNames.append(os.path.join(action_sec_folder, segment[randomIndex]))
                    labels.append(splitNameAction[3])
                    segment.clear()
    print(" COUNT: " + str(count))
    Y = np.array(labels)
    return trainyNames, Y


def pickImage(camera_folder_name, binary=False):
    # pick names and labels
    empty_folder = []
    data = []
    for element in camera_folder_name:
        if os.path.isdir(element):
            split_name_action = element.split("_")
            list_of_frames = natsorted(listdir(element))
            # assert len(list_of_frames) == 16, "problem with seq{} file:{}".format( len(list_of_frames),element)
            list_of_frames = [os.path.join(element, f) for f in list_of_frames]
            data.append([list_of_frames, int(split_name_action[3])])
            x = 1
    print("empty len:{}".format(len(empty_folder)))
    return data


def all_data(Images, exp_folders):
    paths_to_data_train = []
    paths_to_data_test = []
    actions_folders = listdir(exp_folders)
    # print(actions_folders)
    for folder in actions_folders:
        dirS = listdir(os.path.join(exp_folders, folder))
        for element in dirS:
            actions_list = element.split("_")[0]
            action = int(actions_list)
            # function that checks if a performer is in the trainSub. Return False Or True if it is in the trainSub
            u = lambda actions_list: actions_list in Images
            # print(u(Performer))
            if u(actions_list):  # for training
                # print(u(Performer))
                paths_to_data_train.append(os.path.join(exp_folders, folder, element))
            else:  # for testing
                paths_to_data_test.append(os.path.join(exp_folders, folder, element))
    return paths_to_data_train, paths_to_data_test


def process_image(image):
    height, width = (120, 120)
    image = load_img(image, target_size=(height, width))
    imageArray = img_to_array(image)
    x = (imageArray / 255.).astype(np.float32)
    whereAreNans = np.isnan(x)
    x[whereAreNans] = 0
    return x


def ReadFile(ReadFilePath):
    data = []
    labels = []
    with open(ReadFilePath, mode='r') as dataFilesInPath:
        picReader = csv.reader(dataFilesInPath, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for pic in picReader:
            data.append(pic[0])
            labels.append(pic[1])
    labels = np.asarray(labels)
    return data, labels


def WriteFiles(data, labels, savepath):
    with open(savepath, mode='w') as dataFilePaths:
        pictureWrite = csv.writer(dataFilePaths, delimiter='.', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for picture, label in zip(data, labels):
            pictureWrite.writerow([picture, label])


def get_classes(labels):
    return [x for x in range(1, 52)]


def one_hot_labels(classes, labels):
    y = []
    for label in labels:
        y.append(get_class_one_hot(classes, label))

    return y


def get_class_one_hot(classes, class_int):
    label_encoded = classes.index(class_int)
    label_hot = to_categorical(label_encoded, len(classes))
    assert len(label_hot) == len(classes)
    return label_hot
