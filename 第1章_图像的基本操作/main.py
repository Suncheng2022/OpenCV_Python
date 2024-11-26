import cv2
import numpy as np
from urllib import request
from matplotlib import pyplot as plt


# [0105]
# filepath = 'data/瀑布.JPG'
# img = cv2.imread(filepath, flags=1)
# imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# plt.figure(figsize=(8, 7))

# plt.subplot(221)        # Three integers (nrows, ncols, index)
# plt.title('1. RGB')
# plt.axis('off')
# plt.imshow(imgRGB)

# plt.subplot(222)
# plt.title('2. BGR')
# plt.axis('off')
# plt.imshow(img)

# plt.subplot(223)
# plt.title('3. cmap="gray"')
# plt.axis('off')
# plt.imshow(gray, cmap='gray')

# plt.subplot(224)
# plt.title('4. without cmap')
# plt.axis('off')
# plt.imshow(gray)

# plt.tight_layout()
# plt.show()

# [0103]
# filepath = 'data/瀑布.JPG'

# img2 = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), flags=-1)
# saveFile = 'data/瀑布_imencode.jpg'
# cv2.imencode('.jpg', img2)[1].tofile(saveFile)

# img1 = cv2.imread(filepath, flags=1)
# saveFile = 'data/我的瀑布.jpg'
# cv2.imwrite(saveFile, img1)

# [0102]
# response = request.urlopen('https://bkimg.cdn.bcebos.com/pic/b90e7bec54e736d1773da78090504fc2d562690f?x-bce-process=image/format,f_auto/watermark,image_d2F0ZXIvYmFpa2UyNzI,g_7,xp_5,yp_5,P_20/resize,m_lfit,limit_1,h_1080')
# imgUrl = cv2.imdecode(np.asarray(bytearray(response.read()), dtype=np.uint8), -1)

# cv2.imshow('imgUrl', imgUrl)
# key = cv2.waitKey(3000)
# cv2.destroyAllWindows()