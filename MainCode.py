import os
import pandas as pd
from sklearn.model_selection import train_test_split
import Model
import MakingFolders as load
from configparser import ConfigParser
from colorama import Fore

# GIVES THE PATH TO THE CONFIG.INI FILE ON DISK AND ASSIGNED IN ON OBJECT

config_object = ConfigParser()
config_object.read("config.ini")

# ASSIGN DATA FROM CONFIGURATION FILE TO VARABLES IN ORDER TO USES THEM LATER ON
_imageInformation = config_object["IMAGEINFORMATION"]
_neuralDataInformation = config_object["NEURALNETWORK"]
_typeOfExp = config_object["EXP"]

# creation and declaration of file paths in order for data to be retrieved and stored.
mainPath = _imageInformation["picturesFolder"]
dataCreation = _imageInformation["resultsFolder"]
trainPath = os.path.join(mainPath, "actionsL")
testPath = os.path.join(mainPath, "actionsL")
savePath = os.path.join(dataCreation, "Data")
saveModels = os.path.join(dataCreation, "Models")
saveDiagrams = os.path.join(dataCreation, "Diagrams")
exp = _typeOfExp["exp"]
batch = int(_neuralDataInformation["batch"])
inputShape = [20,
              120,
              120,
              3]
weightDecay = _neuralDataInformation["weight_decay"]
epoch = _neuralDataInformation["epoch"]
classesFromIniFile = int(_neuralDataInformation["classes"])
print(Fore.LIGHTGREEN_EX + " =================== PROJECT FILE PATHS THAT NEEDS TO USED ON DISK =======================")
print("", "MAIN PATH --> ", mainPath, "\n",
      "TRAIN PATH --> ", trainPath, "\n",
      "TEST PATH --> ", testPath, "\n",
      "DATA CREATION --> ", dataCreation, "\n",
      "DATA FOLDER --> ", savePath, "\n",
      "SAVE MODELS PATH --> ", saveModels, "\n",
      "SAVE DIAGRAMS PATH --> ", saveDiagrams, "\n",
      "\n============== USING VALUES FROM CONFIG FILE AS BELOW =============\n"
      " EXP --> ", exp, "\n",
      "USING INPUT SHAPE -->", inputShape, "\n",
      "USING BATCH --> ", batch, "\n",
      "USING WEIGHT DECAY --> ", weightDecay, "\n",
      "USING EPOCH --> ", epoch, "\n")
"""time.sleep(5)"""

trainList = os.listdir(trainPath)
testList = os.listdir(testPath)
trainImages, testImages = train_test_split(trainList, test_size=0.15, shuffle=False)
pathsOfTrainData = load.all_data(trainImages, mainPath)
pathsOfTestData = load.all_data(testImages, mainPath)

trainDataX = pathsOfTrainData[1]
testDataY = pathsOfTestData[1]
"""testdataX"""


# DATA CREATION FOLDERS
if not os.path.isdir(dataCreation):
    os.mkdir(os.path.join(dataCreation))
    os.mkdir(os.path.join(savePath))
    os.mkdir(os.path.join(savePath, "Train"))
    os.mkdir(os.path.join(savePath, "Test"))
    os.mkdir(os.path.join(saveDiagrams))
    os.mkdir(os.path.join(saveModels))
    os.mkdir(os.path.join(saveModels, "logs"))
    os.makedirs(os.path.join(saveModels, 'Logs', exp))
    print(
        Fore.LIGHTYELLOW_EX + " ========== DATACREATION FOLDER IS CREATED, BECAUSE NO INSTANCE OF AVAILABLE PATH/S WAS "
                              "FOUND. THIS MESSAGE INDICATES A CREATION OF PATH/S AND IS NOT AN ERROR ========\n",
        "CREATED PATH --> ", dataCreation, "\n",
        "CREATED PATH --> ", savePath, "\n",
        "CREATED PATH --> ", saveModels, "\n",
        "CREATED PATH --> ", saveDiagrams, "\n")
    """time.sleep(5)"""
else:
    print(Fore.LIGHTGREEN_EX + " ========== INSTANCE PATH/S WAS FOUND, THERE IS NO NEED FOR CREATING NEW PATH/S "
                               "=============")

# CHECK IF TRAIN DATA ARE READY. IF NOT CREATE THEM
if os.path.isfile(os.path.join(savePath, "Train", "train.csv")):
    print(Fore.LIGHTGREEN_EX + " ======= TRAIN DATA ARE FOUND IN A FORM OF A CSV FILE =========")
    trainXNames, trainy = load.ReadFile(os.path.join(savePath, "Train", "train.csv"))
else:
    print(Fore.LIGHTYELLOW_EX + " =========== NO CSV FILES FOUND FOR TRAIN DATA, A NEW CSV FILE CALLED TRAIN.CSV IS "
                                "CREATED ===========")
    trainXNames = load.pickImage(trainDataX)
    x, trainy = load.PickingImages(trainPath)
    del x
    print(Fore.LIGHTGREEN_EX + " TOTAL LENGHT: " + str(len(trainXNames)))
    load.WriteFiles(trainXNames, trainy, os.path.join(savePath, "Train", "train.csv"))
    print(Fore.LIGHTGREEN_EX + " ====== PROCESS COMPLETED. TRAIN FILES ARE CREATED ======\n\n\n")

# CHECK IF TEST DATA ARE READY. IF NOT CREATE THEM
if os.path.isfile(os.path.join(savePath, "Test", "test.csv")):
    print(Fore.LIGHTGREEN_EX + " ======= TEST DATA ARE FOUND IN A FORM OF A CSV FILE =========")
    testXnames, testy = load.ReadFile(os.path.join(savePath, "Test", "test.csv"))
else:
    print(Fore.LIGHTYELLOW_EX + " =========== NO CSV FILES FOUND FOR TEST DATA, A NEW CSV FILE CALLED TEST.CSV IS "
                                "CREATED ===========")
    testXnames = load.pickImage(testDataY)
    y, testy = load.PickingImages(testPath)
    del y
    print(Fore.LIGHTGREEN_EX + " TOTAL LENGHT: " + str(len(testXnames)))
    load.WriteFiles(testXnames, testy, os.path.join(savePath, "Test", "test.csv"))
    print(Fore.LIGHTGREEN_EX + " ====== PROCESS COMPLETED. TEST FILES ARE CREATED ======\n\n\n")

trainXNames = pd.DataFrame(trainXNames)
testXnames = pd.DataFrame(testXnames)
trainxNamesSeq = []
testxNamesSeq = []
trainySeq = []
testySeq = []
frames = []
labels = []

for i in range(0, len(trainXNames) // 20):
    frames = trainXNames[i * 20:(i + 1) * 20]
    labels = trainy[i * 20:(i + 1) * 20]
    trainxNamesSeq.append(frames)
    trainySeq.append(labels[0])

for i in range(0, len(testXnames) // 20):
    frames = testXnames[i * 20:(i + 1) * 20]
    labels = testy[i * 20:(i + 1) * 20]
    testxNamesSeq.append(frames)
    testySeq.append(labels[0])

classes = load.get_classes(51)

"""MAYBE: NOT FOR USE UNDER CURRENT IMPLEMENTATION"""
"""trainySeqOneHot = load.one_hot_labels(classes, trainySeq)
testySeqOneHot = load.one_hot_labels(classes, testySeq)"""

trainGenerator = load.DataGenerator(trainxNamesSeq, classes, batch, inputShape, shuffle=True)
testGenerator = load.DataGenerator(testxNamesSeq, classes, batch, inputShape, shuffle=True)

stepsPerEpoch = len(trainxNamesSeq) // int(batch)
valSteps = len(testxNamesSeq) // int(batch)
print("STEPS PER EPOCH --> " + str(stepsPerEpoch))
print("VALIDATION STEPS --> " + str(valSteps))
m = Model.Model(int(batch), weightDecay, stepsPerEpoch, epoch)
model = m.Convolutional3DSimple(51, inputShape)
model.summary()
m.Train(model, trainGenerator, testGenerator, epoch, stepsPerEpoch, valSteps, saveModels, exp)
