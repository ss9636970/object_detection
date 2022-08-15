# object detection

# 1.  training dataset

此任務為物件偵測任務，物件數量為13種物件，下方照片為舉例圖片

![510](https://github.com/ss9636970/object_detection/blob/main/readme/510.jpg)![542](https://github.com/ss9636970/object_detection/blob/main/readme/510.jpg)

資料可分為:

2660張訓練資料

2248張驗證資料



# 2. Single Shot Multibox Detector(SSD)
本篇使用SSD為訓練模型，實作參考自以下github。

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

早期目標檢測任務由兩階段組成，執行物件定位網路以及分類網路。SSD將兩部分封裝為一個網路，從而加快檢測速度。



# 3. 程式碼說明

 objectDetectionModule為SSD模型定義程式碼

funcion.py 為程式中運用到的函式。

dataprocess.ipynb為資料預處理程式碼

main.ipynb為執行模型訓練程式，當中包括讀取資料及模型訓練，以及預測測試資料的程式碼





