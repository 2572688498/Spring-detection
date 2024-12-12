import numpy as np
import cv2
import os
all_data=[]
# 处理图片
def Process_pictures(folder_name, image_name):
    # 输出目录
    output_dir = f'Output Images/{folder_name.split("/")[1]}/'+ folder_name.split("/")[2]
    output_path = output_dir + f'/{image_name.split(".")[0]}.jpg'

    # 读取输入图片
    image = cv2.imread(folder_name + image_name)

    # 输入图片灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 对灰度图片执行高斯滤波
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    #  阈值化
    ret, thresh_basic = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学闭运算：填补空隙，去除噪声
    kernel = np.ones((3, 3), np.uint8)
    img_closed = cv2.morphologyEx(thresh_basic, cv2.MORPH_CLOSE, kernel)

    # 反色处理
    ret, thresh_inv = cv2.threshold(img_closed, 127, 255, cv2.THRESH_BINARY_INV)

    # 在图像中寻找物体轮廓
    cnts, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 循环遍历每一个轮廓
    for c in cnts:
        # 如果当前轮廓的面积太少，认为可能是噪声，直接忽略
        if  cv2.contourArea(c) < 1000 or cv2.contourArea(c) > 1000000:
            continue
        #print(cv2.contourArea(c))
        # 根据物体轮廓计算出外接矩形框
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)

        (width, height) = box[1]
        height_r = round(np.float32(height), 4)
        # 将宽度数据添加到全局数据列表
        all_data.append(f'{image_name}: Width = {height_r}\n')

        # 绘制外接矩形
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 3)

        # 设定新的尺寸
        new_width = int(orig.shape[1] * 0.3)  # 宽度减小
        new_height = int(orig.shape[0] * 0.3)  # 高度减小
        new_size = (new_width, new_height)
        # 使用 INTER_AREA 插值方法缩小图像
        ORIG = cv2.resize(orig, new_size, interpolation=cv2.INTER_AREA)

        # 在图片中绘制结果
        orig = cv2.putText(ORIG, f"The spring width : {height_r:.4f}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
        cv2.imwrite(output_path, orig)

    return output_path  # 返回保存的路径

# 显示图片
def Show_picture(img_path):
    img = cv2.imread(img_path)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 处理文件夹中的所有图片
def Process_all_images_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for image_name in os.listdir(folder_path):
        image_path = Process_pictures(folder_path, image_name)
        #Show_picture(image_path)
    Record_data_in_text(folder_path)
    all_data.clear()

# 记录宽度到文本中
def Record_data_in_text(folder_name):
    arr = [float(x.split("=")[-1]) for x in all_data]
    # 求均值
    average = sum(arr) / len(arr)
    # 保存数据
    with open(f'Output Images/{folder_name.split("/")[1]}/{folder_name.split("/")[2]}/contour_widths.txt', 'a') as f:
        f.writelines(all_data)  # 一次性写入所有数据
        f.write(f"Average: {average}\n")
        f.writelines('--------------------------------------------------------------------------\n')
    # 提示文件保存完成
    print(f"所有宽度数据已保存到 'contour_{folder_name.split('/')[1]}s.txt' 文件中。")

'''
# 文件夹路径
for i in range(1,4):
    folder_name = f'images/width/{i}/'
    # 处理文件夹中所有图像
    Process_all_images_in_folder(folder_name)
'''

