import cv2
import numpy as np
import os
import torch
import logging

from anime_face_detector import create_detector
from train import AnimeSegmentation
from inference import get_mask


def main():
    detector = create_detector('yolov3', device=torch_device)
    path: str = input('Please input file or directory path: ')

    if not os.path.exists('output'):
        os.mkdir('output')

    if os.path.isdir(path):
        for file_path in _list_full_paths(path):
            _detect_face(detector, file_path)
        logging.info('Done!')
    elif os.path.exists(path):
        _detect_face(detector, path)
        logging.info('Done!')
    else:
        logging.error('path not found')
        exit(1)


# https://www.askpython.com/python/examples/python-directory-listing
def _list_full_paths(directory):
    """取得目錄下所有的資料夾並返回完整路徑"""
    list = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            list.append(os.path.join(root, f))
    logging.info(f'path: {directory}')
    logging.info(f'total file count: {len(list)}')
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


def _detect_face(detector, file_path: str):
    """檢測臉部區塊並進行裁切及去背"""
    label_name = os.path.basename(os.path.dirname(file_path))
    file_name, file_extension = os.path.splitext(file_path)
    file_name = os.path.basename(file_name)
    logging.info(f'label name: {label_name}')
    logging.info(f'file name: {file_name}{file_extension}')

    if file_extension == '.db':
        return

    # 使用 np.fromfile 來讀取檔名含有非 ASCII 字元的圖片
    image = cv2.imdecode(np.fromfile(
        file_path, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    if image is None:
        return

    if not os.path.exists(f'output\\{label_name}'):
        os.mkdir(f'output\\{label_name}')

    try:
        faces = detector(image)
    except Exception as ex:
        logging.exception(f'image detector error: {file_path}')
        return

    logging.info(f'face count: {len(faces)}')
    for i in range(len(faces)):
        box = faces[i]['bbox']
        box, score = box[:4], box[4]
        box = np.round(box).astype(int)

        # 忽略低於0.7分的臉
        logging.debug(f'box: {box}, score: {score}')
        if score < 0.7:
            continue

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

        # res = image[max(0, box[1]): min(image.shape[0],  box[3]),
        #            max(0, box[0]): min(image.shape[1],  box[2])]
        res = image[y0: y0 + h, x0: x0 + w, :]
        res = _image_resize(res, 256)

        # 移除背景
        logging.debug('get mask and remove background')
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        mask = get_mask(model, res, use_amp=True, s=256)
        res = np.concatenate(
            (mask * res + 1 - mask, mask * 255), axis=2).astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_RGBA2BGRA)

        # cv2.imencode 轉換成 png 圖檔後使用 tofile 來避免因路徑有非 ASCII 字元導致無法寫入檔案的問題
        cv2.imencode('.png', res)[1].tofile(
            f'output\\{label_name}\\{file_name}_{i}.png')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='[%(asctime)s] %(levelname)s | %(funcName)s | %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    if not os.path.exists('saved_models/isnetis.ckpt'):
        logging.error(
            '"isnetis.ckpt" not found. please download and copy to "saved_models\\isnetis.ckpt"')
        logging.error(
            'model url: https://huggingface.co/skytnt/anime-seg/blob/main/isnetis.ckpt')
        exit(1)

    if torch.cuda.is_available():
        torch_device = 'cuda:0'
    else:
        logging.warn('cuda is unavailable, switch to cpu device')
        torch_device = 'cpu'

    device = torch.device(torch_device)
    model = AnimeSegmentation.try_load(
        'isnet_is', 'saved_models/isnetis.ckpt', torch_device)
    model.eval()
    model.to(device)

    main()
