# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
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

"""hyper parameters"""
use_cuda = True

ROI_SET = True
x1, y1, x2, y2 = 0, 0, 0, 0

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
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


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    # cap = cv2.VideoCapture(0)
    # # cap = cv2.VideoCapture("./test.mp4")
    # cap.set(3, 1280)
    # cap.set(4, 720)
    # print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    mon = {'top': 100, 'left': 100, 'width': 1500, 'height': 1000}
    sct = mss()

    #윈도우 하면 읽어오기
    while True:
        screenshot = sct.grab(mon)
        pic = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)


        # bbox = (500,500,500+1280,500+720) #찍는속도가 느리다
        # pic = ImageG.grab((bbox))

        # pic = pyautogui.screenshot(region=(100,100,1280,720)) #되긴되는데 찍는 속도가 너무 느리다.

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

    # 카메라로 확인하기
    # while True:
    #     ret, img = cap.read()
    #     sized = cv2.resize(img, (m.width, m.height))
    #     sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    #
    #     start = time.time()
    #     boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
    #     finish = time.time()
    #     print('Predicted in %f seconds.' % (finish - start))
    #.
    #     result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)
    #
    #     cv2.imshow('Yolo demo', result_img)
    #     cv2.waitKey(1)
    #
    # cap.release()


def detect_cv2_sim() :
    #소켓통신을 해서 계속 프레임을 받아오고  yolo돌리고 돌린 결과를 다시 통신해서 보내주기
    #우선 프로그램을 실행시킨다.
    out = subprocess.check_output(['../habitat-sim/./build/viewer','--enable-physics', '../habitat-sim/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb'])
    print(out)
    return 0

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-frame', type=int,
                        default=0,
                        help='frame from robot simulation', dest='frame')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        # default='./checkpoints/Yolov4_epoch1.pth',
                        default='./yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        #default='./data/dog.jpg',
                        default=0,
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    #얘의 프레임을 받아와야된다.
    args = get_args()
    args.frame=1
    if args.frame != 0 :
        detect_cv2_sim ()
    elif args.imgfile != 0 :
        detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
