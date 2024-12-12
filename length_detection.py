import numpy as np
import cv2
import os
all_data={}
result_data=[]
lengths = []
widths = []
# 显示图片
def Show_picture(img_path):
    img = cv2.imread(img_path)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 处理图片
def Process_pictures(folder_name, image_name):
    # 输出目录
    output_dir = f'Output Images/length/'+ folder_name.split("/")[2]
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
        #宽度
        height_r = round(np.float32(height), 4)
        #长度
        length_r = round(np.float32(width), 4)

        all_data[image_name] = {"Width": height_r, "Length": length_r}

        # 绘制外接矩形
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 3)

        # 设定新的尺寸
        new_width = int(orig.shape[1] * 0.3)  # 宽度减小
        new_height = int(orig.shape[0] * 0.3)  # 高度减小
        new_size = (new_width, new_height)
        # 使用 INTER_AREA 插值方法缩小图像
        ORIG = cv2.resize(orig, new_size, interpolation=cv2.INTER_AREA)

        # 在图片中绘制结果
        orig = cv2.putText(ORIG, f"width = {height_r:.4f} , length = {length_r:.4f}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)\

        cv2.imwrite(output_path, orig)

    return output_path  # 返回保存的路径

# 处理文件夹中的所有图片
def Process_all_images_in_folder(folder_path):
    lengths.clear()
    widths.clear()
    # 遍历文件夹中的所有文件
    for image_name in os.listdir(folder_path):
        image_path = Process_pictures(folder_path, image_name)
        length = all_data[f'{image_name}']['Length']
        width = all_data[f'{image_name}']['Width']
        lengths.append(length)
        widths.append(width)
        #Show_picture(image_path)

    real_length = lengths[0] + (lengths[1] / widths[1] * widths[0])
    result_data.append(f'length{folder_path.split("/")[-2]} = {real_length}')
    Record_data_in_text(folder_path,real_length)


def averaging(data):
    arr = [float(x.split("=")[1].strip()) for x in data]

    # 求均值
    average = sum(arr) / len(arr)
    return average

# 记录宽度到文本中
def Record_data_in_text(folder_name,data):
    # 保存数据
    with open(f'Output Images/length/{folder_name.split("/")[2]}/contour_length{folder_name.split("/")[2]}.txt', 'a') as f:
        # 一次性写入所有数据
        for key, value in all_data.items():
            f.write(f'{key} = {value}\n')
        f.write(f'real_length={data}\n')
        all_data.clear()




files = os.listdir('images/length')
for i in range(1,9):
    for j, file in enumerate(files, 1):
        folder_name = f'images/length/{i}/{file}/'
        # 处理文件夹中所有图像
        Process_all_images_in_folder(folder_name)

    # 提示文件保存完成
    average = averaging(result_data)
    with open(f'Output Images/length/{i}/contour_length{i}.txt','a') as f:
        f.write(f"Average_real_length: {average}\n")
        f.writelines('--------------------------------------------------------------------------\n')
    print(result_data)
    result_data.clear()
    print(f"所有长度数据已保存到 'contour_length{i}.txt' 文件中。")






