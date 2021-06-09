import logging
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import cv2
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def create_logger(path, log_file):
    # config
    logging.captureWarnings(True)     # 捕捉 py waring message
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    my_logger = logging.getLogger(log_file) #捕捉 py waring message
    my_logger.setLevel(logging.INFO)
    
    # file handler
    fileHandler = logging.FileHandler(path + log_file, 'w', 'utf-8')
    fileHandler.setFormatter(formatter)
    my_logger.addHandler(fileHandler)
    
    # console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    my_logger.addHandler(consoleHandler)
    
    return my_logger

#logger.disabled = True  #暫停 logger
#logger.handlers  # logger 內的紀錄程序
#logger.removeHandler  # 移除紀錄程序
#logger.info('xxx', exc_info=True)  # 紀錄堆疊資訊

def showpic(pic):              #顯示圖片
    cv2.imshow('RGB', pic)     #顯示 RGB 的圖片
    cv2.waitKey(0)             #有這段才不會有bug

def readpic(p):                #讀入圖片
    return cv2.imread(p)
    
def savepic(img, p):           #儲存圖片
    cv2.imwrite(p, img)

def convertLoc(t, d):
    x1, y1, x2, y2 = t[0].item(), t[1].item(), t[2].item(), t[3].item()
    xs = d[0] / 300
    ys = d[1] / 300
    x1, x2 = x1 * ys, x2 * ys
    y1, y2 = y1 * xs, y2 * xs
    return x1, y1, x2, y2


# 處理多張圖片變成 tensor
def pic2tensor(img_list):
    pic = []
    n = len(img_list)
    for i in range(n):
        path = img_list[i]
        image = readpic(path)
        image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
        t = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        t = t.view(1, 3, 300, 300)
        pic.append(t)
    outputs = torch.cat(pic, dim=0)
    return outputs

def data2label(data, labelDict, device):
    objs = data[2:]
    picSize = data[1]
    xp = 300 / picSize[0]
    yp = 300 / picSize[1]
    boxes = []
    Labels = []
    for i in objs:     # 對應到300 * 300
        # x1, y1 = i[1] / picSize[0], i[2] / picSize[1]
        # x2, y2 = i[3] / picSize[0], i[4] / picSize[1]
        x1, y1 = i[1] * xp, i[2] * yp
        x2, y2 = i[3] * xp, i[4] * yp
        box = [x1, y1, x2, y2]
        boxes.append(box)
        Labels.append(labelDict[i[0]])
    return torch.tensor(boxes, dtype=torch.float).to(device), torch.tensor(Labels, dtype=torch.float).to(device)

def dataProcess(datas, labelDict, device):
    picList = []
    boxesList = []
    labelsList = []
    for data in datas:
        picList.append(data[0])
        box, label = data2label(data, labelDict, device)
        boxesList.append(box)
        labelsList.append(label)
    pics = pic2tensor(picList).to(device)
    return pics, boxesList, labelsList

def sumlist(l, n):
    c = 0
    for i in l:
        c += i
    return c / n

def getf1(moduleOutputs, reals):
    pred = torch.argmax(moduleOutputs, dim=1)
    c = f1_score(reals, pred, average='macro')
    return c

def getaccu(moduleOutputs, reals):
    pred = torch.argmax(moduleOutputs, dim=1)
    c = accuracy_score(reals, pred)
    return c

def train(modules, datas, Labels, device):
    module = modules[0]
    lossf = modules[1]
    opt = modules[2]
    inputs, boxes, labels = dataProcess(datas[:1], Labels, device)

    outputs1, outputs2 = module(inputs)
    loss = lossf(outputs1, outputs2, boxes, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

    # if update:
    #     outputs1, outputs2 = module(inputs)
    #     loss = lossf(outputs, labels)
    #     opt.zero_grad()
    #     loss.backward()
    #     opt.step()

    #     return loss.item()
    
    # else:
    #     n = inputs.shape[0]
    #     index = 0
    #     with torch.no_grad():
    #         outputs = []
    #         while index < n:
    #             out = module(inputs[index:index+5, :, :, :])
    #             outputs.append(out)
    #             index += 5
    #         outputs = torch.cat(outputs, dim=0)
    #     return outputs

def tryf():
    print('00000')