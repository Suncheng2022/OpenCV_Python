import cv2
import numpy as np
from urllib import request
from matplotlib import pyplot as plt

# [0108]
img0 = cv2.imread('data/cap_0.jpg')
img1 = cv2.imread('data/cap_1.jpg')
img2 = cv2.imread('data/cap_2.jpg')
img3 = cv2.imread('data/cap_3.jpg')
img4 = cv2.imread('data/cap_4.jpg')
img5 = cv2.imread('data/瀑布.JPG')
imgList = [img0, img1, img2, img3, img4, img5]

saveFile = 'data/imgList.tiff'
assert cv2.imwritemulti(saveFile, imgList), f'写入失败!'
print(f'---- 保存成功 {saveFile}')

rets, imgMulti = cv2.imreadmulti(saveFile)
print(f'---- imgMulti多帧图像长度{len(imgMulti)}')
for i in range(len(imgMulti)):
    print(f'-- {i}/{len(imgMulti)} {imgMulti[i].shape}')
    cv2.imshow(f'{i}', imgMulti[i])
    cv2.waitKey(1000)
cv2.destroyAllWindows()
exit(0)

# [0107]
cap = cv2.VideoCapture(index=0)    # 设置为cv.CAP_DSHOW会报错, 无法正确读取摄像头属性

fps = 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

videoPath = 'data/myvideo.avi'
capWrite = cv2.VideoWriter(filename=videoPath, fourcc=fourcc, fps=fps, frameSize=(width, height))
print(f'capWrite属性: {width}x{height} {fps}, {fourcc}')

sn = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('imshow', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            frame_save_path = f'data/cap_{sn}.jpg'
            cv2.imwrite(frame_save_path, frame)
            sn += 1
            print(f'---- video capture saved in {frame_save_path}')
        elif key == ord('q'):
            break
    else:
        print(f"---- Can't receive frame!")

cap.release()
capWrite.release()
cv2.destroyAllWindows()

# [0106]
# 使用macbook摄像头拍摄一段示例视频
#   使用摄像头, 则必须正确读取设备属性并以此参数设置VideoWriter
# videoCap = cv2.VideoCapture(index=0)
# if videoCap.isOpened():
#     print(f'---- VideoCapture初始化成功!')

#     fps = videoCap.get(cv2.CAP_PROP_FPS)
#     frame_w = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_h = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f'---- Your camera can capture {frame_w}x{frame_h} resolution videos with speed {fps} fps.')

#     video_save_path = 'data/myvideo.avi'
#     videoWrt = cv2.VideoWriter(filename=video_save_path, 
#                                 fourcc=cv2.VideoWriter_fourcc(*'XVID'),    # mpeg-4编码
#                                 fps=fps,
#                                 frameSize=(frame_w, frame_h),
#                                 isColor=True)
#     ret, frame = videoCap.read()
#     frame_count = 0
#     while ret is True:
#         videoWrt.write(frame)
#         ret, frame = videoCap.read()
#         frame_count += 1
#         if frame_count == 200:
#             break
#     videoWrt.release()
#     videoCap.release()

# videoRead = 'data/myvideo.avi'
# capRead = cv2.VideoCapture(filename=videoRead)

# width = int(capRead.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(capRead.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = capRead.get(cv2.CAP_PROP_FPS)
# frameCount = int(capRead.get(cv2.CAP_PROP_FRAME_COUNT))
# print(f'---- {width}x{height} {fps} {frameCount}')

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# capWrite = cv2.VideoWriter(filename='data/writeVideo.avi', fourcc=fourcc, fps=fps, frameSize=(width, height))

# frameNum = 0
# timef = 30
# while capRead.isOpened():
#     ret, frame = capRead.read()
#     if ret:
#         frameNum += 1
#         if frameNum % timef == 0:
#             cv2.imshow(f'frame_{frameNum}', frame)
#             capWrite.write(frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         print(f'capRead.read() failed!')
#         break
# capRead.release()
# capWrite.release()
# cv2.destroyAllWindows()

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