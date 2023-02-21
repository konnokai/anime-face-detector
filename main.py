import cv2
import numpy as np
import os
import torch

from anime_face_detector import create_detector
from train import AnimeSegmentation
from inference import get_mask

detector = create_detector('yolov3')


def main():
    if not os.path.exists('output'):
        os.mkdir('output')

    path: str = input('Please input file or directory path: ')
    if os.path.isdir(path):
        for file_path in _list_full_paths(path):
            _detect_face(file_path)
        print('Done!')
    elif os.path.exists(path):
        _detect_face(path)
        print('Done!')
    else:
        print('Error: path not found')


# https://www.askpython.com/python/examples/python-directory-listing
def _list_full_paths(directory):
    """取得目錄下所有的資料夾並返回完整路徑"""
    list = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            list.append(os.path.join(root, f))
    print(f'path: {directory}')
    print(f'total file count: {len(list)}')
    return list


# https://stackoverflow.com/a/44659589
def _image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """使用 OpenCV 來縮放圖片"""
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def _detect_face(file_path: str):
    """檢測臉部區塊並進行裁切及去背"""
    label_name = os.path.basename(os.path.dirname(file_path))
    file_name, file_extension = os.path.splitext(file_path)
    file_name = os.path.basename(file_name)
    print(f'label name: {label_name}')
    print(f'file name: {file_name}{file_extension}')

    if file_extension == '.db':
        return

    image = cv2.imdecode(np.fromfile(
        file_path, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    if image is None:
        return

    if not os.path.exists(f'output\\{label_name}'):
        os.mkdir(f'output\\{label_name}')

    try:
        faces = detector(image)
    except Exception as ex:
        print(f'image detector error: {file_path}')
        print(ex)
        return

    print(f'face count: {len(faces)}')
    for i in range(len(faces)):
        box = faces[i]['bbox']
        box, score = box[:4], box[4]
        box = np.round(box).astype(int)

        # 縮放75%以獲得更多細節
        # https://zhuanlan.zhihu.com/p/44179128
        x = box[0]
        y = box[1]
        w = box[2] - x
        h = box[3] - y
        cx = x + w // 2
        cy = y + h // 2
        x0 = cx - int(0.75 * w)
        x1 = cx + int(0.75 * w)
        y0 = cy - int(0.75 * h)
        y1 = cy + int(0.75 * h)

        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 >= image.shape[1]:
            x1 = image.shape[1] - 1
        if y1 >= image.shape[0]:
            y1 = image.shape[0] - 1
        w = x1 - x0
        h = y1 - y0
        if w > h:
            x0 = x0 + w // 2 - h // 2
            x1 = x1 - w // 2 + h // 2
            w = h
        else:
            y0 = y0 + h // 2 - w // 2
            y1 = y1 - h // 2 + w // 2
            h = w

        # 忽略低於0.7分的臉
        print(f'box: {box}, score: {score}')
        if score < 0.7:
            continue

        # res = image[max(0, box[1]): min(image.shape[0],  box[3]),
        #            max(0, box[0]): min(image.shape[1],  box[2])]
        res = image[y0: y0 + h, x0: x0 + w, :]
        res = _image_resize(res, 256)

        # 移除背景
        print('get mask')
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        mask = get_mask(model, res, use_amp=True, s=256)
        res = np.concatenate(
            (mask * res + 1 - mask, mask * 255), axis=2).astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_RGBA2BGRA)

        cv2.imencode('.png', res)[1].tofile(
            f'output\\{label_name}\\{file_name}_{i}.png')


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = AnimeSegmentation.try_load(
        'isnet_is', 'saved_models/isnetis.ckpt', 'cuda:0')
    model.eval()
    model.to(device)

    main()
