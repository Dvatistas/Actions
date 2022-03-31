import numpy as np
import shutil
from os import listdir, path, mkdir, makedirs
from natsort import natsorted


def CreatePathsToData(dataFolder):
    pathsToData = []
    folders = natsorted(listdir(dataFolder))
    print(dataFolder)
    for folder in folders:

