import torch
import time
import cv2
import numpy as np
import os
from PIL import Image
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("/home/dongpeijie/deep_sort_pytorch-master/uolov3")

from models import *
from utils.datasets import *
from utils.utils import *


class InferYOLOv3(object):
    def __init__(self,
                 cfg,
                 img_size,
                 weight_path,
                 data_cfg,
                 device,
                 conf_thres=0.3,
                 nms_thres=0.5):
        self.device = device
        self.cfg = cfg
        self.img_size = img_size
        self.weight_path = weight_path
        # self.img_file = img_file
        self.model = Darknet(cfg).to(device)
        self.model.load_state_dict(
            torch.load(weight_path, map_location=device)['model'])
        self.model.to(device).eval()
        self.class_names = load_classes(parse_data_cfg(data_cfg)['names'])
        self.colors = [random.randint(0, 255) for _ in range(3)]
        # self.class_names = self.load_class_names(namesfile)

        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    def predict(self, ori_img):
        # singleDataloader = LoadSingleImages(img_file, img_size=img_size)
        # path, img, ori_img = singleDataloader.__next__()

        resized_img, _, _, _ = letterbox(ori_img, new_shape=self.img_size)

        resized_img = resized_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        resized_img = np.ascontiguousarray(
            resized_img, dtype=np.float32)  # uint8 to float32
        resized_img /= 255.0

        # TODO: how to get img and ori_img

        resized_img = torch.from_numpy(resized_img).unsqueeze(0).to(
            self.device)
        pred, _ = self.model(resized_img)
        det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(resized_img.shape[2:], det[:, :4],
                                      ori_img.shape).round()

            # Print results to screen
            print('%gx%g ' % resized_img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, self.class_names[int(c)]), end=', ')

            resized_img = np.array(resized_img.cpu())
            # Draw bounding boxes and labels of detections

            bboxes, confs, cls_confs, cls_ids = [], [], [], []

            for *xyxy, conf, cls_conf, cls_id in det:
                label = '%s %.2f' % (self.class_names[int(cls_id)], conf)
                bboxes.append(xyxy)
                confs.append(conf)
                cls_confs.append(cls_conf)
                cls_ids.append(cls_id)
                # plot_one_box(xyxy, ori_img, label=label, color=self.colors)
            return np.array(bboxes), np.array(cls_confs), np.array(cls_ids)
        else:
            return None, None, None

    # def load_class_names(self, namesfile):
    #     with open(namesfile, 'r', encoding='utf8') as fp:
    #         class_names = [line.strip() for line in fp.readlines()]
    #     return class_names

    def __call__(self, img_file):
        singleDataloader = LoadSingleImages(img_file, img_size=self.img_size)
        path, img, im0 = singleDataloader.__next__()

        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        pred, _ = self.model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                      im0.shape).round()

            # Print results to screen
            # print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                # print('%g %ss' % (n, self.class_names[int(c)]), end=', ')

            img = np.array(img.cpu())
            # Draw bounding boxes and labels of detections

            bboxes, confs, cls_confs, cls_ids = [], [], [], []

            for *xyxy, conf, cls_conf, cls_id in det:
                label = '%s %.2f' % (self.class_names[int(cls_id)], conf)
                bboxes.append(xyxy)
                confs.append(conf)
                cls_confs.append(cls_conf)
                cls_ids.append(cls_id)
                # plot_one_box(xyxy, im0, label=label, color=self.colors)
        
        # cv2.imwrite("./save_predict.jpg", im0)
        return np.array(bboxes), np.array(cls_confs), np.array(cls_ids)


    # def plot_bbox(self, ori_img, boxes):
    #     img = ori_img
    #     height, width = img.shape[:2]
    #     for box in boxes:
    #         # get x1 x2 x3 x4
    #         x1 = int(round(((box[0] - box[2] / 2.0) * width).item()))
    #         y1 = int(round(((box[1] - box[3] / 2.0) * height).item()))
    #         x2 = int(round(((box[0] + box[2] / 2.0) * width).item()))
    #         y2 = int(round(((box[1] + box[3] / 2.0) * height).item()))
    #         cls_conf = box[5]
    #         cls_id = box[6]
    #         # import random
    #         # color = random.choices(range(256),k=3)
    #         color = [int(x) for x in np.random.randint(256, size=3)]
    #         # put texts and rectangles
    #         img = cv2.putText(img, self.class_names[cls_id], (x1, y1),
    #                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    #         img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    #     return img

    # def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    #     # Plots one bounding box on image img
    #     tl = line_thickness or round(
    #         0.002 * max(img.shape[0:2])) + 1  # line thickness
    #     color = color or [random.randint(0, 255) for _ in range(3)]
    #     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    #     cv2.rectangle(img, c1, c2, color, thickness=tl)
    #     if label:
    #         tf = max(tl - 1, 1)  # font thickness
    #         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3,
    #                                  thickness=tf)[0]
    #         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #         cv2.rectangle(img, c1, c2, color, -1)  # filled
    #         cv2.putText(img,
    #                     label, (c1[0], c1[1] - 2),
    #                     0,
    #                     tl / 3, [225, 255, 255],
    #                     thickness=tf,
    #                     lineType=cv2.LINE_AA)


if __name__ == "__main__":
    #################################################
    cfg = './cfg/yolov3-1cls.cfg'
    img_size = 416
    weight_path = './weights/yolov3-1cls/best.pt'
    img_file = "./images/train2014/0137-2112.jpg"
    data_cfg = "./data/voc_small.data"
    conf_thres = 0.3
    nms_thres = 0.5
    device = torch_utils.select_device()
    #################################################
    yolo = InferYOLOv3(cfg, img_size, weight_path, data_cfg, device)
    bbox_xcycwh, cls_conf, cls_ids = yolo(img_file)
    print(bbox_xcycwh.shape, cls_conf.shape, cls_ids.shape)

    img = cv2.imread(img_file)
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = img
    print(im.shape)
    bbox_xcycwh, cls_conf, cls_ids = yolo.predict(im)
    print(bbox_xcycwh.shape, cls_conf.shape, cls_ids.shape)

# import torch
# # import time
# import numpy as np
# import cv2

# from darknet import Darknet
# from yolo_utils import get_all_boxes, nms, plot_boxes_cv2

# class YOLOv3(object):
#     def __init__(self,
#                  cfgfile,
#                  weightfile,
#                  namesfile,
#                  use_cuda=True,
#                  is_plot=False,
#                  is_xywh=False,
#                  conf_thresh=0.3,
#                  nms_thresh=0.4):
#         # net definition
#         self.net = Darknet(cfgfile)
#         self.net.load_weights(weightfile)
#         print('Loading weights from %s... Done!' % (weightfile))
#         self.device = "cuda" if use_cuda else "cpu"
#         self.net.eval()
#         self.net.to(self.device)

#         # constants
#         self.size = self.net.width, self.net.height
#         self.conf_thresh = conf_thresh
#         self.nms_thresh = nms_thresh
#         self.use_cuda = use_cuda
#         self.is_plot = is_plot
#         self.is_xywh = is_xywh
#         self.class_names = self.load_class_names(namesfile)

#     def __call__(self, ori_img):
#         # img to tensor
#         assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
#         img = ori_img.astype(np.float) / 255.

#         img = cv2.resize(img, self.size)
#         img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
#         # forward
#         with torch.no_grad():
#             img = img.to(self.device)
#             out_boxes = self.net(img)
#             boxes = get_all_boxes(out_boxes, self.conf_thresh,
#                                   self.net.num_classes, self.use_cuda)[0]
#             boxes = nms(boxes, self.nms_thresh)
#             # print(boxes)
#         # plot boxes
#         if self.is_plot:
#             return self.plot_bbox(ori_img, boxes)
#         if len(boxes) == 0:
#             return None, None, None

#         height, width = ori_img.shape[:2]
#         boxes = np.vstack(boxes)
#         bbox = np.empty_like(boxes[:, :4])
#         if self.is_xywh:
#             # bbox x y w h
#             bbox[:, 0] = boxes[:, 0] * width
#             bbox[:, 1] = boxes[:, 1] * height
#             bbox[:, 2] = boxes[:, 2] * width
#             bbox[:, 3] = boxes[:, 3] * height
#         else:
#             # bbox xmin ymin xmax ymax
#             bbox[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * width
#             bbox[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * height
#             bbox[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * width
#             bbox[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * height
#         cls_conf = boxes[:, 5]
#         cls_ids = boxes[:, 6]
#         return bbox, cls_conf, cls_ids

#     def load_class_names(self, namesfile):
#         with open(namesfile, 'r', encoding='utf8') as fp:
#             class_names = [line.strip() for line in fp.readlines()]
#         return class_names

#     def plot_bbox(self, ori_img, boxes):
#         img = ori_img
#         height, width = img.shape[:2]
#         for box in boxes:
#             # get x1 x2 x3 x4
#             x1 = int(round(((box[0] - box[2] / 2.0) * width).item()))
#             y1 = int(round(((box[1] - box[3] / 2.0) * height).item()))
#             x2 = int(round(((box[0] + box[2] / 2.0) * width).item()))
#             y2 = int(round(((box[1] + box[3] / 2.0) * height).item()))
#             cls_conf = box[5]
#             cls_id = box[6]
#             # import random
#             # color = random.choices(range(256),k=3)
#             color = [int(x) for x in np.random.randint(256, size=3)]
#             # put texts and rectangles
#             img = cv2.putText(img, self.class_names[cls_id], (x1, y1),
#                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#             img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         return img

# if __name__ == '__main__':
#     yolo3 = YOLOv3("cfg/yolo_v3.cfg",
#                    "yolov3.weights",
#                    "cfg/coco.names",
#                    is_plot=True)
#     print("yolo3.size =", yolo3.size)
#     import os
#     root = "../demo"
#     files = [os.path.join(root, file) for file in os.listdir(root)]
#     files.sort()
#     for filename in files:
#         img = cv2.imread(filename)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         res = yolo3(img)
#         # save results
#         # cv2.imwrite("../result/{}".format(os.path.basename(filename)),res[:,:,(2,1,0)])
#         # imshow
#         # cv2.namedWindow("yolo3", cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow("yolo3", 600,600)
#         # cv2.imshow("yolo3",res[:,:,(2,1,0)])
#         # cv2.waitKey(0)
