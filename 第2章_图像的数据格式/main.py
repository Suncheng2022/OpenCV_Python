import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def main():

    # [0208] LUT 函数查表实现颜色缩减
    filepath = 'data/cap_0.jpg'
    gray = cv2.imread(filepath, 0)
    h, w = gray.shape[:2]

    timeBegin = cv2.getTickCount()
    imgGray32 = np.empty((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            imgGray32[i][j] = (gray[i][j] // 8) * 8
    timeEnd = cv2.getTickCount()
    time = (timeEnd - timeBegin) / cv2.getTickFrequency()
    print(f'Grayscale reduction by for loop: {time:.4f} s')

    timeBegin = cv2.getTickCount()
    table32 = np.array([(i // 8) * 8 for i in range(256)]).astype(np.uint8)
    gray32 = cv2.LUT(gray, table32)
    timeEnd = cv2.getTickCount()
    time = (timeEnd - timeBegin) / cv2.getTickFrequency()
    print(f'Grayscale reduction by LUT: {time:.4f} s')

    table8 = np.array([(i // 32) * 32 for i in range(256)]).astype(np.uint8)
    gray8 = cv2.LUT(gray, table8)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131)
    plt.axis('off')
    plt.title('1. Gray-256')
    plt.imshow(gray, cmap='gray')

    plt.subplot(132)
    plt.axis('off')
    plt.title('2. Gray-32')
    plt.imshow(gray32, cmap='gray')

    plt.subplot(133)
    plt.axis('off')
    plt.title('3. Gray-8')
    plt.imshow(gray8, cmap='gray')

    plt.tight_layout()
    plt.show()

    # [0207] LUT 函数查表实现图像反转
    # filepath = 'data/cap_0.jpg'
    # img = cv2.imread(filepath, 1)
    # h, w, ch = img.shape

    # timeBegin = cv2.getTickCount()
    # imgInv = np.empty((h, w, ch), np.uint8)
    # for i in range(h):
    #     for j in range(w):
    #         for k in range(ch):
    #             imgInv[i][j][k] = 255 - img[i][j][k]
    # timeEnd = cv2.getTickCount()
    # print(f'for循环实现图像颜色翻转, 耗时: {(timeEnd - timeBegin) / cv2.getTickFrequency():.4f}')

    # timeBegin = cv2.getTickCount()
    # transTable = np.array([[255 - i] for i in range(256)]).astype(np.uint8)
    # invLUT = cv2.LUT(img, transTable)
    # timeEnd = cv2.getTickCount()
    # print(f'LUT实现图像颜色翻转, 耗时: {(timeEnd - timeBegin) / cv2.getTickFrequency():.4f}')

    # timeBegin = cv2.getTickCount()
    # subtract = 255 - img
    # timeEnd = cv2.getTickCount()
    # time = (timeEnd - timeBegin) / cv2.getTickFrequency()
    # print(f'直接相减实现图像颜色翻转, 耗时: {(timeEnd - timeBegin) / cv2.getTickFrequency():.4f}')

    # [0206]马赛克
    # filepath = 'data/cap_0.jpg'
    # img = cv2.imread(filepath, 1)

    # roi = cv2.selectROI(img, showCrosshair=True, fromCenter=False)
    # x, y, wRoi, hRoi = roi
    # imgROI = img[y:y + hRoi, x:x + wRoi].copy()
    # print(x, y, wRoi, hRoi)

    # plt.figure(figsize=(9, 6))
    # plt.subplot(231)
    # plt.title('1. Original')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # plt.subplot(232)
    # plt.title('2. ROI')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(imgROI, cv2.COLOR_BGR2RGB))

    # masaic = np.zeros(imgROI.shape, np.uint8)
    # ksize = [5, 10, 20]
    # for i in range(3):
    #     k = ksize[i]
    #     for h in range(0, hRoi, k):
    #         for w in range(0, wRoi, k):
    #             color = imgROI[h, w]
    #             masaic[h:h + k, w:w + k, :] = color
    #     imgMasaic = img.copy()
    #     imgMasaic[y:y + hRoi, x:x + wRoi] = masaic
    #     plt.subplot(2, 3, i + 4)
    #     plt.title(f'Coding image (size={k})')
    #     plt.axis('off')
    #     plt.imshow(cv2.cvtColor(imgMasaic, cv2.COLOR_BGR2RGB))
    # plt.subplot(233)
    # plt.title('3. Mosaic')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(masaic, cv2.COLOR_BGR2RGB))
    # plt.show()


    # [0205]
    # filepath = 'data/瀑布.JPG'
    # img = cv2.imread(filepath, 1)

    # h, w = 20, 10
    # pxBGR = img[h, w]
    # print(f'(1) img[{h}][{w}] = {pxBGR}')

    # for ch in range(3):
    #     print(f'(2) {img[h, w, ch]}', end=' ')
    # print()
    
    # for ch in range(3):
    #     print(f'(3) {img.item(h, w, ch)}', end='  ')
    # print()
    
    # print(f'(4) old img[h, w] = {img[h, w]}')
    # img[h, w, :] = 255
    # print(f'(4) new img[h, w] = {img[h, w]}')

    # [0204]
    # filepath = 'data/瀑布.JPG'
    # img = cv2.imread(filepath)
    # bImg, gImg, rImg = cv2.split(img)
    # imgMerge = cv2.merge([bImg, gImg, rImg])
    # imgStack = np.stack((bImg, gImg, rImg), axis=2)
    # cv2.imshow('imgMerge', imgMerge)
    # cv2.imshow('imgStach', imgStack)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # imgB = img.copy()
    # imgB[..., 1] = 0
    # imgB[..., 2] = 0

    # imgG = img.copy()
    # imgG[..., 0] = 0
    # imgG[..., 2] = 0

    # imgR = img.copy()
    # imgR[..., 0] = 0
    # imgR[..., 1] = 0

    # imgGR = img.copy()
    # imgGR[..., 0] = 0

    # plt.figure(figsize=(8, 7))

    # plt.subplot(221)
    # plt.title('1. B channel')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB))

    # plt.subplot(222)
    # plt.title('2. G channel')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(imgG, cv2.COLOR_BGR2RGB))

    # plt.subplot(223)
    # plt.title('3.R channel')
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))

    # plt.subplot(224)
    # plt.title('4. GR channel')
    # plt.imshow(cv2.cvtColor(imgGR, cv2.COLOR_BGR2RGB))
    # plt.tight_layout()
    # plt.show()

    # [0203]
    # filepath = 'data/瀑布.JPG'
    # img = cv2.imread(filepath, 1)
    # xmin, ymin, w, h = 180, 190, 200, 200
    # imgCrop = img[ymin:ymin + h, xmin:xmin + w].copy()
    # print(f'img.shape:{img.shape}, imgCrop.shape:{imgCrop.shape}')

    # logo = cv2.imread('data/Nike.png')
    # imgH1 = cv2.resize(img, (400, 400))
    # imgH2 = cv2.resize(logo, (300, 400))
    # imgH3 = imgH2.copy()
    # stachH = cv2.hconcat((imgH1, imgH2, imgH3))
    # print(f'imgH1.shape {imgH1.shape}\n'
    #       f'imgH2.shape {imgH2.shape}\n'
    #       f'imgH3.shape {imgH3.shape}\n'
    #       f'stackH.shape {stachH.shape}')
    # plt.figure(figsize=(9, 4))      # w, h
    # plt.imshow(cv2.cvtColor(stachH, cv2.COLOR_BGR2RGB))
    # plt.xlim(0, 900)
    # plt.ylim(400, 0)
    # plt.show()

    # imgV1 = cv2.resize(img, (400, 400))
    # imgV2 = cv2.resize(logo, (400, 300))
    # stackV = cv2.vconcat((imgV1, imgV2))
    # print(f'imgV1.shape {imgV1.shape}\n'
    #       f'imgV2.shape {imgV2.shape}\n'
    #       f'stachV.shape {stackV.shape}\n')
    
    # cv2.imshow('DemoStachH', stachH)
    # cv2.imshow('DemoStackV', stackV)
    # key = cv2.waitKey(0)
    
    # [0202]
    # height, width, ch = 400, 300, 3

    # imgEmpty = np.empty(shape=(height, width, ch), dtype=np.uint8)
    # imgBlack = np.zeros(shape=(height, width, ch), dtype=np.uint8)
    # imgWhite = np.ones(shape=(height, width, ch), dtype=np.uint8) * 255

    # img = cv2.imread('data/瀑布.JPG', 1)
    # imgBlackLike = np.zeros_like(img)
    # imgWhiteLike = np.ones_like(img) * 255

    # randomByteArray = bytearray(os.urandom(height * width * ch))
    # flatArray = np.array(randomByteArray)
    # print(f'---- flatArray.shape {flatArray.shape}')
    # imgRGBrand1 = flatArray.reshape(width, height, ch)
    # imgRGBrand2 = flatArray.reshape(height, width, ch)

    # grayWhite = np.ones((height, width), np.uint8) * 255
    # grayBlack = np.zeros((height, width), np.uint8)
    # grayEye = np.eye(width)
    # randomByteArray = bytearray(os.urandom(height * width))
    # flatArray = np.array(randomByteArray)
    # imgGrayrand = flatArray.reshape(height, width)

    # img1 = img.copy()
    # img1[:, :, :] = 0
    # print(f'img1 is equal to img? --> {img1 is img}')

    # img2 = img
    # img2[:, :, :] = 0
    # print(f'img2 is equal to img? --> {img2 is img}')

    # [0201]
    # filepath = 'data/瀑布.JPG'
    # img = cv2.imread(filepath, 1)
    # gray = cv2.imread(filepath, 0)

    # print(f'Ndim of img and gray: {img.ndim} {gray.ndim}')
    # print(f'Shape of img and gray: {img.shape} {gray.shape}')
    # print(f'Size of img and gray: {img.size} {gray.size}')

    # imgFloat = img.astype(np.float32) / 255
    # print(f'{imgFloat.dtype}, {img.dtype}, {gray.dtype}')

if __name__ == "__main__":
    main()