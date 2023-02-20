import cv2
import numpy as np
import os

from anime_face_detector import create_detector

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
    list = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            list.append(os.path.join(root, f))
    print(f'path: {directory}')
    print(f'total file count: {len(list)}')
    return list


def _detect_face(file_path: str):
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

    if not os.path.exists(f'output\{label_name}'):
        os.mkdir(f'output\{label_name}')

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

        print(f'box: {box}, score: {score}')
        if score < 0.7:
            continue

        res = image[max(0, box[1]): min(image.shape[0],  box[3]),
                    max(0, box[0]): min(image.shape[1],  box[2])]
        cv2.imencode(file_extension, res)[1].tofile(f'output\{label_name}\{file_name}_{i}{file_extension}')


if __name__ == '__main__':
    main()