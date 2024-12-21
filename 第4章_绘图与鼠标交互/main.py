import cv2
import numpy as np
from matplotlib import pyplot as plt

# [0407]
img1 = cv2.imread('../images/Lena.tif')
img2 = img1.copy()

text = 'Digital Image Processing, suncheng01@xxx.com'
fontList = [cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            cv2.FONT_ITALIC]
fontScale = .8
color = [255, 255, 255]
for i in range(len(fontList)):
    pos = (10, 40 * (i + 1))
    cv2.putText(img1, text, pos, fontList[i], fontScale, color)

from PIL import Image, ImageDraw, ImageFont
imgPIL = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
string = 'PIL 添加中文字体 \nby suncheng'
pos = (50, 20)
color = (0, 0, 255)     # RGB
textSize = 50
drawPIL = ImageDraw.Draw(imgPIL)
fontText = ImageFont.truetype('楷体.ttf', textSize, encoding='utf-8')
drawPIL.text(pos, string, color, font=fontText)
imgText = cv2.cvtColor(np.asarray(imgPIL), cv2.COLOR_RGB2BGR)

plt.figure(figsize=(9, 3.5))
plt.subplot(121), plt.title('1. cv.putText'), plt.axis('off'), plt.imshow(img1[..., ::-1])
plt.subplot(122), plt.title('2. PIL.Image'), plt.axis('off'), plt.imshow(imgText[..., ::-1])
plt.show()
exit(-1)

# [0403]
height, width, channels = 300, 400, 3
img = np.ones((height, width, channels), np.uint8) * 192

cx, cy, w, h = (200, 150, 200, 100)
img1 = img.copy()
cv2.circle(img1, (cx, cy), 4, (0, 0, 255), -1)
angle = [15, 30, 45, 60, 75, 90]
# box = np.zeros((4, 2), np.int32)
for i in range(len(angle)):
    rect = ((cx, cy), (w, h), angle[i])
    box = np.int32(cv2.boxPoints(rect))
    color = (30 * i, 0, 255 - 30 * i)
    cv2.drawContours(img1, [box], 0, color, 1)
    print(rect)

x, y, w, h = (200, 100, 160, 100)
img2 = img.copy()
cv2.circle(img2, (x, y), 4, (0, 0, 255), -1)
angle = [15, 30, 45, 60, 75, 90, 120, 150, 180, 225]
for i in range(len(angle)):
    ang = angle[i]
    x1, y1 = x, y
    x2 = int(x + w * np.cos(ang))
    y2 = int(y + w * np.sin(ang))
    x3 = int(x + w * np.cos(ang) - h * np.sin(ang))
    y3 = int(y + w * np.sin(ang) + h * np.cos(ang))
    x4 = int(x - h * np.sin(ang))
    y4 = int(y + h * np.cos(ang))
    color = (30 * i, 0, 255 - 30 * i)
    box = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    cv2.drawContours(img2, [box], 0, color, 1)
plt.figure(figsize=(9, 3.2))
plt.subplot(121), plt.title(f'1. img1'), plt.axis('off'), plt.imshow(img1[..., ::-1])
plt.subplot(122), plt.title(f'2. img2'), plt.axis('off'), plt.imshow(img2[..., ::-1])
plt.tight_layout()
plt.show()

exit(-1)

# [0402]
height, width, channels = 300, 200, 3
img = np.ones((height, width, channels), np.uint8) * 192

img1 = img.copy()
cv2.rectangle(img1, (0, 80), (100, 220), (0, 0, 255), 2)
cv2.rectangle(img1, (80, 0), (220, 100), (0, 255, 0), 2)        # 同上比较, 坐标点由(x, y)写为(y, x)
cv2.rectangle(img1, (150, 120), (400, 200), 255, 2)
cv2.rectangle(img1, (50, 10), (100, 50), (128, 0, 0), 1)
cv2.rectangle(img1, (150, 10), (200, 50), (192, 0, 0), 2)
cv2.rectangle(img1, (250, 10), (300, 50), (255, 0, 0), 4)
cv2.rectangle(img1, (50, 250), (100, 290), (128, 0, 0), -1)
cv2.rectangle(img1, (150, 250), (200, 290), (192, 0, 0), -1)
cv2.rectangle(img1, (250, 250), (300, 290), (255, 0, 0), -1)

img2 = img.copy()
x, y, w, h = (50, 100, 200, 100)
cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
text = f'{x}, {y}, {w}, {h}'
cv2.putText(img2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255))

img3 = np.zeros((height, width), np.uint8)
cv2.line(img3, (0, 40), (320, 40), 64, 2)
cv2.line(img3, (0, 80), (320, 80), (128, 128, 255), 2)
cv2.line(img3, (0, 120), (320, 120), (192, 64, 255), 2)
cv2.rectangle(img3, (20, 250), (50, 220), 128, -1)
cv2.rectangle(img3, (80, 250), (110, 210), (128, 0, 0), -1)
cv2.rectangle(img3, (140, 250), (170, 200), (128, 255, 255), -1)
cv2.rectangle(img3, (200, 250), (230, 190), 192, -1)
cv2.rectangle(img3, (260, 250), (290, 180), 255, -1)

plt.figure(figsize=(9, 3.3))
plt.subplot(131), plt.title('1. img1'), plt.axis('off'), plt.imshow(img1[..., ::-1])
plt.subplot(132), plt.title('2. img2'), plt.axis('off'), plt.imshow(img2[..., ::-1])
plt.subplot(133), plt.title('3. img3'), plt.axis('off'), plt.imshow(img3, cmap='gray')
plt.tight_layout()
plt.show()

exit(-1)

# [0401]
height, width, channels = 180, 200, 3
img = np.ones(shape=(height, width, channels), dtype=np.uint8) * 160

img1 = img.copy()           # cv2的绘图是就地操作
cv2.line(img1, (0, 0), (200, 180), (0, 0, 255), 1)
cv2.line(img1, (0, 0), (100, 180), (0, 255, 0), 1)
cv2.line(img1, (0, 40), (200, 40), (128, 0, 0), 2)
cv2.line(img1, (0, 80), (200, 80), 128, 2)
cv2.line(img1, (0, 120), (200, 120), 255, 2)

img2 = img.copy()
cv2.line(img2, (20, 50), (180, 10), (255, 0, 0), 1, cv2.LINE_8)
cv2.line(img2, (20, 90), (180, 50), (255, 0, 0), 1, cv2.LINE_AA)

cv2.line(img2, (20, 130), (180, 90), (255, 0, 0), cv2.LINE_8)
cv2.line(img2, (20, 170), (180, 130), (255, 0, 0), cv2.LINE_AA)

img3 = img.copy()
cv2.arrowedLine(img3, (20, 20), (180, 20), (0, 0, 255), tipLength=.05)       # 似乎不必接受返回值--就地操作
cv2.arrowedLine(img3, (20, 60), (180, 60), (0, 0, 255), tipLength=.1)
cv2.arrowedLine(img3, (20, 100), (180, 100), (0, 0, 255), tipLength=.15)
cv2.arrowedLine(img3, (180, 100), (20, 100), (0, 0, 255), tipLength=.15)
cv2.arrowedLine(img3, (20, 140), (210, 140), (0, 0, 255), tipLength=.2)

img4 = cv2.line(img, (0, 100), (150, 100), (0, 255, 0), 1)       # 水平线段
img5 = cv2.line(img, (75, 0), (75, 200), (0, 0, 255), 1)         # 竖直线段

img6 = np.zeros((height, width), np.uint8)              # 灰度图像为二维数组
cv2.line(img6, (0, 10), (200, 10), (0, 255, 255), 2)    # 灰度图, 颜色则只有第一个通道有效
cv2.line(img6, (0, 30), (200, 30), (64, 128, 255), 2)
cv2.line(img6, (0, 60), (200, 60), (128, 64, 255), 2)
cv2.line(img6, (0, 100), (200, 100), (255, 0, 255), 2)
cv2.line(img6, (20, 0), (20, 200), 128, 2)
cv2.line(img6, (60, 0), (60, 200), (255, 0, 0), 2)
cv2.line(img6, (100, 0), (100, 200), (255, 255, 255), 2)
print(img6.shape, img6.shape)

plt.figure(figsize=(9, 6))
plt.subplot(231), plt.title('1. img1'), plt.axis('off'), plt.imshow(img1[..., ::-1])
plt.subplot(232), plt.title('2. img2'), plt.axis('off'), plt.imshow(img2[..., ::-1])
plt.subplot(233), plt.title('3. img3'), plt.axis('off'), plt.imshow(img3[..., ::-1])
plt.subplot(234), plt.title('4. img4'), plt.axis('off'), plt.imshow(img4[..., ::-1])
plt.subplot(235), plt.title('5. img5'), plt.axis('off'), plt.imshow(img5[..., ::-1])
plt.subplot(236), plt.title('6. img6'), plt.axis('off'), plt.imshow(img6, cmap='gray')
plt.tight_layout()
plt.show()

exit(-1)