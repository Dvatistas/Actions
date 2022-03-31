import os

dir_path = '/DataCreation'
try:
    os.rmdir(dir_path)
except OSError as e:
    print("Error: %s : %s" % (dir_path, e.strerror))
