import cv2
import numpy as np
from os.path import join
from matplotlib import pyplot as plt

# 教材使用的图片
images_dir = r'/Users/sunchengcheng/Projects/OpenCV_Python/images'


# 3.5调节图像的色彩平衡
# [0306]
img = cv2.imread(join(images_dir, 'Lena.tif'), 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 单通道LUT, 形状为(256, )
k = .6          # 色彩拉伸系数
lutWeaken = np.array([int(k * i) for i in range(256)]).astype('uint8')
lutEqual = np.array([i for i in range(256)]).astype('uint8')
lutRaisen = np.array([int(255 * (1 - k) + i * k) for i in range(256)]).astype('uint8')      # 作用大概是拉升低像素值的影响
# 多通道LUT, 调节饱和度, 形状为(1, 256, 3)
lutSWeaken = np.dstack((lutEqual, lutWeaken, lutEqual))
lutSRaisen = np.dstack((lutEqual, lutRaisen, lutEqual))
# 多通道LUT, 调节明度
lutVWeaken = np.dstack((lutEqual, lutEqual, lutWeaken))
lutVRaisen = np.dstack((lutEqual, lutEqual, lutRaisen))

blendSWeaken = cv2.LUT(hsv, lutSWeaken)
blendSRaisen = cv2.LUT(hsv, lutSRaisen)
blendVWeaken = cv2.LUT(hsv, lutVWeaken)
blendVRaisen = cv2.LUT(hsv, lutVRaisen)

plt.figure(figsize=(9, 6))

plt.subplot(231)
plt.axis('off')
plt.title('1. Saturation weaken')
plt.imshow(cv2.cvtColor(blendSWeaken, cv2.COLOR_HSV2RGB))

plt.subplot(232)
plt.axis('off')
plt.title('2. Original saturation')
plt.imshow(img[..., ::-1])

plt.subplot(233)
plt.axis('off')
plt.title('3. Saturation raisen')
plt.imshow(cv2.cvtColor(blendSRaisen, cv2.COLOR_HSV2RGB))

plt.subplot(234)
plt.axis('off')
plt.title('4. Value weaken')
plt.imshow(cv2.cvtColor(blendVWeaken, cv2.COLOR_HSV2RGB))

plt.subplot(235)
plt.axis('off')
plt.title('5. Original value')
plt.imshow(img[..., ::-1])

plt.subplot(236)
plt.axis('off')
plt.title('6. Value raisen')
plt.imshow(cv2.cvtColor(blendVRaisen, cv2.COLOR_HSV2RGB))

plt.tight_layout()
plt.show()
exit(-1)

# [0305]使用多通道LUT调节色彩平衡
img = cv2.imread(join(images_dir, 'Lena.tif'), 1)

# 单通道LUT, 形状(256, )
maxG = 128
lutHalf = np.array([int(i * maxG / 255) for i in range(256)]).astype('uint8')       # 0~255 映射到 0~128
lutEqual = np.array([i for i in range(256)]).astype('uint8')                        # 0~255 映射到 0~255

# 多通道LUT, 要求形状必须为(1, 256, 3).
# np.dstack() 会将[M, N]的2D-array或[M, ]的1D-array转为 [M, N, 1]或[1, M, 1]
lut3HalfB = np.dstack((lutHalf, lutEqual, lutEqual))        # 用于衰减B通道
lut3HalfG = np.dstack((lutEqual, lutHalf, lutEqual))        # 用于衰减G通道
lut3HalfR = np.dstack((lutEqual, lutEqual, lutHalf))        # 用于衰减R通道

# 使用多通道LUT进行颜色替换, 此处功能实现通道的颜色衰减
blendHalfB = cv2.LUT(img, lut3HalfB)
blendHalfG = cv2.LUT(img, lut3HalfG)
blendHalfR = cv2.LUT(img, lut3HalfR)
print(f'img.shape: {img.shape}, blendHalfB.shape: {blendHalfB.shape}, blendHalfG.shape:{blendHalfG.shape}, blendHalfR.shape:{blendHalfR.shape}')

plt.figure(figsize=(9, 3.5))            # 参数figsize=(w, h) 单位inch

plt.subplot(131)
plt.axis('off')
plt.title('1.B_ch half decayed')
plt.imshow(blendHalfB[..., ::-1])

plt.subplot(132)
plt.axis('off')
plt.title('2.G_ch half decayed')
plt.imshow(blendHalfG[..., ::-1])

plt.subplot(133)
plt.axis('off')
plt.title('3.R_ch half decayed')
plt.imshow(blendHalfR[..., ::-1])

plt.tight_layout()
plt.show()
exit(-1)

# 3.4图像的色彩风格滤镜
img = cv2.imread(join(images_dir, 'Fig0301.png'), 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

plt.figure(figsize=(9, 6))
plt.subplot(231)
plt.title('origin')
plt.axis('off')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

from matplotlib import cm
cmList = ['cm.copper', 'cm.hot', 'cm.YlOrRd', 'cm.rainbow', 'cm.prism']
for i in range(len(cmList)):
    cmMap = eval(cmList[i])(np.arange(256))     # 返回[256, 4], RGBA, 0~1
    lutC3 = np.zeros((1, 256, 3))
    lutC3[0, :, 0] = np.array(cmMap[:, 2] * 255).astype(np.uint8)
    lutC3[0, :, 1] = np.array(cmMap[:, 1] * 255).astype(np.uint8)
    lutC3[0, :, 2] = np.array(cmMap[:, 0] * 255).astype(np.uint8)

    cmLUTC3 = cv2.LUT(img, lutC3).astype('uint8')
    print(img.shape, cmMap.shape, lutC3.shape)
    plt.subplot(2, 3, i + 2)
    plt.axis('off')
    plt.title(f'{i + 2}. {cmList[i]}')
    plt.imshow(cv2.cvtColor(cmLUTC3, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()
exit(-1)

# 3.3多模态数据合成的伪彩色图像
# [0303] 利用多光谱编码合成彩色幸运
composite = cv2.imread(join(images_dir, 'Fig0303.png'), 1)      # NASA合成图像
grayOpti = cv2.imread(join(images_dir, 'Fig0303a.jpg'), 0)      # optical
grayXray = cv2.imread(join(images_dir, 'Fig0303b.jpg'), 0)      # Xray
grayInfr = cv2.imread(join(images_dir, 'Fig0303c.jpg'), 0)      # Infrared 红外线的

h, w = grayOpti.shape[:2]
# 伪彩色处理
pseudoXray = cv2.applyColorMap(grayXray, colormap=cv2.COLORMAP_TURBO)       # 漩涡
pseudoOpti = cv2.applyColorMap(grayOpti, colormap=cv2.COLORMAP_MAGMA)       # 岩浆
pseudoInfr = cv2.applyColorMap(grayInfr, colormap=cv2.COLORMAP_HOT)
plt.figure(figsize=(9, 6))
plt.subplot(131), plt.title('pseudoXray'), plt.axis('off'), plt.imshow(cv2.cvtColor(pseudoXray, cv2.COLOR_BGR2RGB))
plt.subplot(132), plt.title('pseudoOpti'), plt.axis('off'), plt.imshow(cv2.cvtColor(pseudoOpti, cv2.COLOR_BGR2RGB))
plt.subplot(133), plt.title('pseudoInfr'), plt.axis('off'), plt.imshow(cv2.cvtColor(pseudoInfr, cv2.COLOR_BGR2RGB))

# 多光谱编码合成
compose1 = np.zeros((h, w, 3), np.uint8)
compose1[..., 0] = grayOpti
compose1[..., 1] = grayXray
compose1[..., 2] = grayInfr

compose2 = np.zeros((h, w, 3), np.uint8)
compose2[..., 0] = grayXray
compose2[..., 1] = grayOpti
compose2[..., 2] = grayInfr

plt.figure(figsize=(9, 6))      # 参数figsize=(w, h)  单位是inch

plt.subplot(231)
plt.axis('off')
plt.title('1.CrabNebla-Xray')
plt.imshow(grayXray, cmap='gray')

plt.subplot(232)
plt.axis('off')
plt.title('2.CrabNebla-Optical')
plt.imshow(grayOpti, cmap='gray')

plt.subplot(233)
plt.axis('off')
plt.title('3.CrabNebla-Infrared')
plt.imshow(grayInfr, cmap='gray')

plt.subplot(234)
plt.axis('off')
plt.title('4.Composite pseudo1')
plt.imshow(cv2.cvtColor(compose1, cv2.COLOR_BGR2RGB))

plt.subplot(235)
plt.axis('off')
plt.title('5.Composite pseudo2')
plt.imshow(cv2.cvtColor(compose2, cv2.COLOR_BGR2RGB))

plt.subplot(236)
plt.axis('off')
plt.title('6.Composite by NASA')
plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
exit(-1)

# 3.2 灰度图像的伪彩色处理
# 伪彩色图像: 对单色图像进行处理, 重建为彩色图像的效果
# [0302] 灰度图像转换为伪彩色图像
gray = cv2.imread(join(images_dir, 'Fig0301.png'), 0)
h, w = gray.shape[:2]

pseudo1 = cv2.applyColorMap(gray, colormap=cv2.COLORMAP_HOT)
pseudo2 = cv2.applyColorMap(gray, colormap=cv2.COLORMAP_PINK)
pseudo3 = cv2.applyColorMap(gray, colormap=cv2.COLORMAP_RAINBOW)
pseudo4 = cv2.applyColorMap(gray, colormap=cv2.COLORMAP_HSV)
pseudo5 = cv2.applyColorMap(gray, colormap=cv2.COLORMAP_TURBO)

plt.figure(figsize=(9, 6))      # Width, height in inches

plt.subplot(231)
plt.axis('off')
plt.title('1. Gray')
plt.imshow(gray, cmap='gray')

plt.subplot(232)
plt.axis('off')
plt.title('2. COLORMAP_HOT')
plt.imshow(cv2.cvtColor(pseudo1, cv2.COLOR_BGR2RGB), cmap='gray')

plt.subplot(233)
plt.axis('off')
plt.title('3. COLORMAP_PINK')
plt.imshow(cv2.cvtColor(pseudo2, cv2.COLOR_BGR2RGB))

plt.subplot(234)
plt.axis('off')
plt.title('4. COLOR_RAINBOW')
plt.imshow(cv2.cvtColor(pseudo3, cv2.COLOR_BGR2RGB))

plt.subplot(235)
plt.axis('off')
plt.title('5. COLORMAP_HSV')
plt.imshow(cv2.cvtColor(pseudo4, cv2.COLOR_BGR2RGB))

plt.subplot(236)
plt.axis('off')
plt.title('6. COLORMAP_TURBO')
plt.imshow(cv2.cvtColor(pseudo5, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
exit(-1)

# [0301]
imgBGR = cv2.imread(join(images_dir, 'Lena.tif'), 1)

imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
imgYCrCb = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2YCrCb)
imgHLS = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HLS)
imgXYZ = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2XYZ)
imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
imgYUV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2YUV)

titles = ['BGR', 'RGB', 'GRAY', 'HSV', 'YCrCb', 'HLS', 'XYZ', 'LAB', 'YUV']
images = [imgBGR, imgRGB, imgGray, imgHSV, imgYCrCb, imgHLS, imgXYZ, imgLAB, imgYUV]

plt.figure(figsize=(10, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f'{i + 1}.{titles[i]}')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()