# -*- coding: utf-8 -*-
'''
@Time          : 21/02/18 14:50
@Author        : yunsujeon
@File          : dev1.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
from PIL import ImageGrab
from PIL import Image
import cv2
import numpy as np
import pyautogui
from mss import mss
import subprocess
import cv2

"""hyper parameters"""
use_cuda = True
ROI_SET = True
x1, y1, x2, y2 = 0, 0, 0, 0


def load_network(cfgfile, weightfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    if use_cuda:
        m.cuda()
    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)
    return m, class_names


##############################################################

##############################################################

def detect_cv2_img(imgfile, m, class_names):
    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


##############################################################

##############################################################

def detect_cv2_crop(m, class_names):
    mon = {'top': 100, 'left': 100, 'width': 1500, 'height': 1000}
    sct = mss()

    print(m.width, m.height)
    # 윈도우 하면 읽어오기
    while True:
        screenshot = sct.grab(mon)
        pic = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)

        image = np.array(pic)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        sized = cv2.resize(image, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(image, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)

        key = cv2.waitKey(100)
        if key == ord("q"):
            print("Quit")
            break

    cv2.destroyAllWindows()


##############################################################

##############################################################

def detect_cv2_webcam(m, class_names):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


##############################################################

##############################################################

def detect_cv2_sim_vid(m, class_names):
    if args.record == 1:
        subprocess.run(["python", "../habitat-sim/interaction.py"])

    cap = cv2.VideoCapture('../habitat-sim/output/fetch.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        sized = cv2.resize(frame, (m.width, m.height))
        # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)  # ndarray 를 numpy로 해서 yolo들어가게 해준다.

        result_img = plot_boxes_cv2(frame, boxes[0], savename=None, class_names=class_names)
        cv2.imshow('Yolo demo', result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    return 0


##############################################################

##############################################################


def detect_cv2_sim_frame(m,class_names) :
    # 구현중.. C++ API 사용
    return 0


##############################################################

##############################################################

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-simvid', type=int,
                        default=0,
                        help='video frame from robot simulation', dest='simvid')
    parser.add_argument('-record', type=int, default=0,
                        help='want create video=1', dest='record')
    parser.add_argument('-imgfile', type=str, default=0,
                        # default='./data/dog.jpg',
                        help='path of your image file.', dest='imgfile')
    parser.add_argument('-webcam', type=int, default=0,  # 웹캠인풋
                        help='get frame use webcam', dest='webcam')
    parser.add_argument('-simframe', type=int, default=0,  # 시뮬레이션 실시간
                        help='video frame from realtim robot simulation API', dest='simframe')
    parser.add_argument('-crop', type=int, default=0,  # 화면 크롭
                        help='get crop frame on display', dest='crop')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        # default='./checkpoints/Yolov4_epoch1.pth',
                        default='./yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    m, class_names = load_network(args.cfgfile, args.weightfile)
    if args.simvid != 0:
        detect_cv2_sim_vid(m, class_names)
    elif args.imgfile != 0:
        detect_cv2_img(args.imgfile, m, class_names)
    elif args.webcam != 0:
        detect_cv2_webcam(m, class_names)
    elif args.simframe != 0:
        detect_cv2_sim_frame(m, class_names)
    elif args.crop != 0:
        detect_cv2_crop(m, class_names)
