import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=8589934592) # minimum disk space required in Byte
    cache = {}
    cnt = 1
    for i in list(range(nSamples)):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def read_text(path):
    with open(path) as f:
        text = f.read()
    text = text.strip() # remove leading and trailing whitespace

    return text


def create_train_set():
    outputPath = 'dataset/train/'

    filenames = [os.path.splitext(f)[0] for f in glob.glob("data_train/*.jpg")]
    jpg_files = [s + ".jpg" for s in filenames]
    imgLabelLists = []
    for p in jpg_files:
        try:
            imgLabelLists.append((p, read_text(p.replace('.jpg', '.txt'))))
        except:
            continue

    # imgLabelList = [ (p, read_text(p.replace('.jpg', '.txt'))) for p in imagePathList]
    # sort labelList by length of label
    imgLabelList = sorted(imgLabelLists, key=lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    txtLists = [p[1] for p in imgLabelList]

    createDataset(outputPath, imgPaths, txtLists, lexiconList=None, checkValid=True)


def create_val_set():
    outputPath = 'dataset/val/'

    filenames = [os.path.splitext(f)[0] for f in glob.glob("data_valid/*.jpg")]
    jpg_files = [s + ".jpg" for s in filenames]
    imgLabelLists = []
    for p in jpg_files:
        try:
            imgLabelLists.append((p, read_text(p.replace('.jpg', '.txt'))))
        except:
            continue

    # imgLabelList = [ (p, read_text(p.replace('.jpg', '.txt'))) for p in imagePathList]
    # sort labelList by length of label
    imgLabelList = sorted(imgLabelLists, key=lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    txtLists = [p[1] for p in imgLabelList]

    createDataset(outputPath, imgPaths, txtLists, lexiconList=None, checkValid=True)

if __name__ == '__main__':
    create_train_set()
    create_val_set()
