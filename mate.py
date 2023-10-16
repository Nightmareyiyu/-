import cv2

# 加载眼镜图片
glasses = cv2.imread('glasses.jpg', -1)

# 打开笔记本内置摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while (cap.isOpened()):
    ret, frame = cap.read()  # 从摄像头中实时读取视频
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转换图像为灰度格式，以便进行人脸检测
    #cv2.imshow("img", frame)  # 显示最终处理的效果

    def overlay_img(img, img_over, img_over_x, img_over_y):
        """
        覆盖图像
        :param img: 背景图像
        :param img_over: 覆盖的图像
        :param img_over_x: 覆盖图像在背景图像上的横坐标
        :param img_over_y: 覆盖图像在背景图像上的纵坐标
        :return: 两张图像合并之后的图像
        """
        img_h, img_w, img_p = img.shape  # 背景图像宽、高、通道数
        img_over_h, img_over_w, img_over_c = img_over.shape  # 覆盖图像高、宽、通道数
        if img_over_c == 3:  # 通道数小于等于3
            img_over = cv2.cvtColor(img_over, cv2.COLOR_BGR2BGRA)  # 转换成4通道图像
        for w in range(0, img_over_w):  # 遍历列
            for h in range(0, img_over_h):  # 遍历行
                if img_over[h, w, 3] != 0:  # 如果不是全透明的像素
                    for c in range(0, 3):  # 遍历三个通道
                        x = img_over_x + w  # 覆盖像素的横坐标
                        y = img_over_y + h  # 覆盖像素的纵坐标
                        if x >= img_w or y >= img_h:  # 如果坐标超出最大宽高
                            break  # 不做操作
                        img[y, x, c] = img_over[h, w, c]  # 覆盖像素
        return img  # 完成覆盖的图像


    height, width, channel = glasses.shape  # 获取眼镜图像高、宽、通道数
    # 加载级联分类器
    face_cascade = cv2.CascadeClassifier("D:\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # 识别人脸
    for (x, y, w, h) in faces:# 遍历所有人脸的区域
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        gw = w  # 眼镜缩放之后的宽度
        gh = int(height * w / width)  # 眼镜缩放之后的高度度
        glass_img = cv2.resize(glasses, (gw, gh))  # 按照人脸大小缩放眼镜
        overlay_img(frame, glass_img, x, y + int(h * 1 / 4))  # 将眼镜绘制到人脸上


    cv2.imshow("Video", frame)  # 在窗口中显示视频
    k = cv2.waitKey(1)  # 图像的刷新时间为1毫秒
    if k == 32:  # 按下空格键
        cap.release()  # 关闭笔记本内置摄像头
        cv2.destroyWindow("Video")  # 销毁名为Video的窗口
        cv2.imwrite("copy.png", frame)  # 保存按下空格键时摄像头视频中的图像
        cv2.imshow('img', frame)  # 显示按下空格键时摄像头视频中的图像
        #cv2.waitKey()  # 刷新图像
        break
cv2.destroyAllWindows()  # 销毁显示图像的窗口
