import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import glob
import os
import csv
import cv2
import models.crnn as crnn

DATA_DIRECTORY = '/home/ubuntu/imagedata/'


TXT_RESULT_FLD = DATA_DIRECTORY + 'SROIE2019_transcript/' + 'txt_result/'
TXT_PROCESSED_FLD = DATA_DIRECTORY + 'SROIE2019_transcript/' + 'txt_processed_result/'
BB_RESULT_FLD = DATA_DIRECTORY + 'SROIE2019_transcript/' + 'bb_result/'
RESULT_DRAW_FLD = DATA_DIRECTORY + 'SROIE2019_transcript/' + 'result_draw/'
IMAGE_FOLDER = DATA_DIRECTORY + 'SROIE2019/' + 'img_contrast/*.*' 
BB_FOLDER = DATA_DIRECTORY + 'SROIE2019/' + 'label/'
BB_INT_FOLDER = DATA_DIRECTORY + 'SROIE2019/' + 'label_new/'

def convert_bbratio_to_int():
    for img in  glob.glob(IMAGE_FOLDER):
        image = cv2.imread(img)
        H,W,C = image.shape
        words_list =[]
        with open(BB_FOLDER + os.path.basename(img).split('.')[0]+'.txt', 'r') as boxes:
            for line in csv.reader(boxes):
                #box = [int(string, 10) for string in line[0:8]]
                box = line[0:8]
                words = [line[-1]]
                intbox = [str(int(float(box[0])*W)), ',', str(int(float(box[1])*H)),',', str(int(float(box[2])*W)),',', str(int(float(box[3])*H)),',', str(int(float(box[4])*W)),',', str(int(float(box[5])*H)),',', str(int(float(box[6])*W)),',', str(int(float(box[7])*H)),',']
                
                intbox.extend(words)
                words_list.append(intbox)
        with open(BB_INT_FOLDER + os.path.basename(img).split('.')[0]+'.txt', 'w+') as resultfile:
            for line in words_list:
                resultfile.writelines(line + ['\n'])



def predict_this_box(image, model, alphabet):
    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((200, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-30s => %-30s' % (raw_pred, sim_pred))
    return sim_pred


def load_images_to_predict():
    # load model
    model_path = './expr/netCRNN_199_423.pth'
    alphabet = '0123456789,.:(%$!^&-/);<~|`>?+=_[]{}"\'@#*ABCDEFGHIJKLMNOPQRSTUVWXYZ\ '
    imgH = 32 # should be 32
    nclass = len(alphabet) + 1
    nhiddenstate = 256

    model = crnn.CRNN(imgH, 1, nclass, nhiddenstate)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path).items()})

    # load image

    #img = [os.path.basename(f) for f in glob.glob(IMAGE_FOLDER)]
    #filenames, extns = [os.path.splitext(f)[0:1] for f in glob.glob(IMAGE_FOLDER)]
    #exts = [os.path.splitext(f)[1] for f in glob.glob(IMAGE_FOLDER)]
    #img_files = [s + e for s , e in (filenames, extns)]
    for img in  glob.glob(IMAGE_FOLDER):
        image = Image.open(img).convert('L')
        words_list = []
        #with open(BB_FOLDER + img.split('/')[1].split('.')[0]+'.txt', 'r') as boxes:
        print('Ïmage name:', os.path.basename(img))
        with open(BB_INT_FOLDER + os.path.basename(img).split('.')[0]+'.txt', 'r') as boxes:
            for line in csv.reader(boxes):
                box = [int(string, 10) for string in line[0:8]]
                boxImg = image.crop((box[0], box[1], box[4], box[5]))
                words = predict_this_box(boxImg, model, alphabet)
                words_list.append(words)
        with open( TXT_RESULT_FLD + os.path.basename(img).split('.')[0] +'.txt', 'w+') as resultfile:
            for line in words_list:
                resultfile.writelines(line+'\n')


def process_txt():
    filenames = [os.path.splitext(f)[0] for f in glob.glob(TXT_RESULT_FLD + "*.txt")]
    old_files = [s + ".txt" for s in filenames]
    for old_file in old_files:
        new = []
        with open(old_file, "r") as old:
            for line in csv.reader(old):
                if not line:
                    continue
                if not line[0]:
                    continue
                if line[0][0] == ' ' or line[0][-1] == ' ':
                    line[0] = line[0].strip()
                if ' ' in line[0]:
                    line = line[0].split(' ')
                new.append(line)
        #with open(TXT_PROCESSED_FLD + old_file.split('/')[-1], "w+") as newfile:
        with open(TXT_PROCESSED_FLD + os.path.basename(old_file), "w+") as newfile:
            wr = csv.writer(newfile, delimiter = '\n')
            new = [[s[0].upper()] for s in new]
            wr.writerows(new)


def for_task3():
    filenames = [os.path.splitext(f)[0] for f in glob.glob( BB_INT_FOLDER + "*.txt")]
    box_files = [s + ".txt" for s in filenames]
    for boxfile in box_files:
        box = []
        with open(boxfile,'r') as boxes:
            for line in csv.reader(boxes):
                box.append([int(string, 10) for string in line[0:8]])
        words = []
        #with open(TXT_PROCESSED_FLD + os.path.basename(boxfile), 'r') as prediction:
        with open(TXT_RESULT_FLD + os.path.basename(boxfile), 'r') as prediction:
            for line in csv.reader(prediction):
                words.append(line)
        words = [s if len(s)!=0 else [' '] for s in words]
        new = []
        for line in zip(box,words):
            a,b = line
            new.append(a+b)
        with open(BB_RESULT_FLD + os.path.basename(boxfile), 'w+') as newfile:
            csv_out = csv.writer(newfile)
            for line in new:
                csv_out.writerow(line)


def draw():
    filenames = [os.path.splitext(f)[0] for f in glob.glob(BB_RESULT_FLD + "*.txt")]
    txt_files = [s + ".txt" for s in filenames]
    for txt in txt_files:
        #image = cv2.imread('test_original/'+ txt.split('/')[1].split('.')[0]+'.jpg', cv2.IMREAD_COLOR)
        image = cv2.imread( IMAGE_FOLDER + os.path.basename(txt).split('.')[0] +'.jpg', cv2.IMREAD_COLOR)
        os.path.basename(txt).split('.')[0]
        with open(txt, 'r') as txt_file:
            for line in csv.reader(txt_file):
                box = [int(string, 10) for string in line[0:8]]
                if len(line) < 9:
                    print(txt)
                cv2.rectangle(image, (box[0], box[1]), (box[4], box[5]), (0,255,0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, line[8].upper(), (box[0],box[1]), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite('task2_result_draw/'+ txt.split('/')[1].split('.')[0]+'.jpg', image)

def draw_new():
    for img in  glob.glob(IMAGE_FOLDER):
        image = cv2.imread(img,cv2.IMREAD_COLOR)
        #H,W,C = image.shape
        #words_list =[]
        with open(BB_RESULT_FLD + os.path.basename(img).split('.')[0]+'.txt', 'r') as txt_file:
            for line in csv.reader(txt_file):
                box = [int(string, 10) for string in line[0:8]]
                if len(line) < 9:
                    print(txt)
                cv2.rectangle(image, (box[0], box[1]), (box[4], box[5]), (0,255,0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, line[8].upper(), (box[0],box[1]), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(RESULT_DRAW_FLD + os.path.basename(img), image)



if __name__ == "__main__":
    #convert_bbratio_to_int()
    load_images_to_predict()
    process_txt()
    for_task3()
    #draw()
    draw_new()
