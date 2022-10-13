import argparse
import sys
import torch
import cv2
import time
import os
import numpy as np

sys.path.insert(0, './YOLOX')

from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.utils import postprocess
from utils.visualize import vis
from YOLOX.yolox.utils.visualize import plot_tracking
from YOLOX.yolox.tracker.byte_tracker import BYTETracker
from torch2trt import TRTModule
from YOLOX.yolox.exp.build import get_exp_by_name
from loguru import logger


COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo")
    parser.add_argument("demo", default="image",
                        help="demo type, eg. image, video and webcam")
    parser.add_argument(
        "--path", default="/home/ds1/tuyennt/YOLOX-ByteTrack-Car-Counter/videos/1.mp4", help="path to video")
    parser.add_argument("--camid", default=0, help="webcam demo camera id")
    parser.add_argument("--track_thresh", type=float,
                        default=0.4, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float,
                        default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float,
                        default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False,
                        action="store_true", help="test mot20.")
    parser.add_argument("--tsize", default=None,
                        type=int, help="test img size")
    parser.add_argument("-n", "--name", type=str,
                        default="yolox-s", help="model name")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        default="output",
        help="whether to save the inference result of image/video",
    )

    return parser


class Predictor():
    def __init__(self, model=None):
        super(Predictor, self).__init__()

        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        print("Device = ", self.device)
        self.cls_names = COCO_CLASSES

        self.preproc = ValTransform(legacy=False)
        self.exp = get_exp_by_name(model)
        self.exp.test_size = (640, 640)
        print("Test_size = ", self.exp.test_size)

        self.test_size = self.exp.test_size
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        # checkpoint = torch.load(ckpt, map_location="cpu")
        # self.model.load_state_dict(checkpoint["model"])

        self.trt_file = "./YOLOX/YOLOX_outputs/yolox_s/model_trt.pth"
        self.model.head.decode_in_inference = False
        self.decoder = self.model.head.decode_outputs

        self.load_modelTRT()

    def load_modelTRT(self):
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(self.trt_file))
        x = torch.ones(
            1, 3, self.exp.test_size[0], self.exp.test_size[1]).cuda()
        self.model(x)
        self.model = model_trt

    def detect(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0],
                    self.test_size[1] / img.shape[1])
        img, _ = self.preproc(img, None, self.test_size)

        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.exp.num_classes,  self.exp.test_conf, self.exp.nmsthre,
                class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        info = {}
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            info['boxes'], info['scores'], info['class_ids'], info['box_nums'] = None, None, None, 0
            return img, info

        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]

        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)

        info['boxes'] = bboxes
        info['scores'] = scores
        info['class_ids'] = cls
        info['box_nums'] = output.shape[0]

        return vis_res, info


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def check_traffic_light(img):
    global traffic_color
    if isinstance(img, str):
        img = cv2.imread(img)
    # output = output.cpu()
    # cls_ids = output[:, 6]
    # boxes = output[:, 0:4]
    # boxes /= ratio
    # for i in range(len(boxes)):
    #     cls_id = int(cls_ids[i])
    #     scores = output[:, 4] * output[:, 5]
    #     score = scores[i]
    #     if score < 0.4:
    #         continue
    #     if cls_id == 9:
    #         box = boxes[i]
    #         x0 = int(box[0])
    #         if x0 < 0:
    #             x0 = 0
    #         y0 = int(box[1])
    #         if y0 < 0:
    #             y0 = 0
    #         x1 = int(box[2])
    #         if x1 < 0:
    #             x1 = 0
    #         y1 = int(box[3])
    #         if y1 < 0:
    #             y1 = 0

    box = [602.3,   0, 620.52,  50.92]
    x0 = int(box[0]/0.25)
    y0 = int(box[1]/0.25)
    x1 = int(box[2]/0.25)
    y1 = int(box[3]/0.25)

    light = img[y0:y1, x0:x1]
    light_hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)

    mask_red_1 = cv2.inRange(light_hsv, (0, 100, 100), (10, 255, 255))
    mask_red_2 = cv2.inRange(light_hsv, (160, 100, 100), (180, 255, 255))
    mask_red = cv2.add(mask_red_1, mask_red_2)
    mask_green = cv2.inRange(light_hsv, (40, 50, 50), (90, 255, 255))
    mask_yellow = cv2.inRange(light_hsv, (15, 150, 150), (35, 255, 255))

    num_red = np.count_nonzero(mask_red == 255)
    num_green = np.count_nonzero(mask_green == 255)
    num_yellow = np.count_nonzero(mask_yellow == 255)

    if num_red > num_green and num_red > num_yellow:
        traffic_color = "red"
    elif num_green > num_red and num_green > num_yellow:
        traffic_color = "green"
    elif num_yellow > num_red and num_yellow > num_green:
        traffic_color = "yellow"
    else:
        traffic_color = "unknown"

    return traffic_color


def get_area_detect(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    pts = np.array([[0, 1920], [0, 888], [640, 350], [2500, 350], [2500, 1920]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    pts_2 = np.array(
        [[450, 540], [640, 400], [2380, 400], [2380, 550]], np.int32)
    pts_2 = pts_2.reshape((-1, 1, 2))
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # img = cv2.polylines(img, [pts_2], isClosed=True, color=(0, 0, 255), thickness=2)
    img_detect = cv2.bitwise_and(img, img, mask=mask)

    return img_detect


def image_demo(predictor, exp, args):
    if os.path.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    save_folder = "/home/ds1/tuyennt/ByteTrack_Car/output/images"
    files.sort()
    object_id = 0
    results = []
    filter_class = [2, 3]
    tracker = BYTETracker(args)
    for image_name in files:
        base_name = os.path.basename(image_name)
        txt = base_name.split('.')[0] + '.txt'
        save_file = os.path.join(save_folder, txt)

        img = cv2.imread(image_name)
        img_detect = get_area_detect(img)
        traffic_color = check_traffic_light(img)
        outputs, img_info = predictor.detect(img_detect)
        # result_frame,info = predictor.visual(outputs[0], img_info)
        if outputs[0] is not None:
            online_targets = tracker.update(
                outputs[0], [img_info['height'], img_info['width']], exp.test_size, filter_class)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

                    if traffic_color == "red":
                        y = tlwh[1]
                        if y <= 550:
                            img = cv2.line(img, (0, 300), (2560, 300), (0, 255, 0), thickness=2)

                    results.append(
                        f"{object_id} {float(tlwh[0])} {float(tlwh[1])} {float(tlwh[2])} {float(tlwh[3])}\n")
            online_im = plot_tracking(
                img, online_tlwhs, online_ids, traffic_color=traffic_color)

            box_light = [602.3,   0, 620.52,  50.92]
            # box /= 0.25
            x0 = int(box_light[0]/0.25)
            y0 = int(box_light[1]/0.25)
            x1 = int(box_light[2]/0.25)
            y1 = int(box_light[3]/0.25)

            cv2.rectangle(online_im, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(save_folder, base_name), online_im)

        # with open(save_file, 'w') as f:
        #     f.writelines(results)


def video_demo(predictor, exp, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.demo == "video":
        save_folder_video = os.path.join(
            args.save_result, os.path.basename(args.path.split(".")[0]))
        os.makedirs(save_folder_video, exist_ok=True)
        save_path_video = os.path.join(
            save_folder_video, "output_" + args.path.split("/")[-1])
        res_file = os.path.join(save_folder_video, "output_" +
                                os.path.basename(args.path.split(".")[0] + ".txt"))
    else:
        save_path_video = "output/camera/camera.mp4"
        res_file = "output/camera/camera.txt"
    logger.info(f"video save_path is {save_path_video}")
    vid_writer = cv2.VideoWriter(
        save_path_video, cv2.VideoWriter_fourcc(
            *"mp4v"), fps, (int(width), int(height))
    )

    tracker = BYTETracker(args, frame_rate=22)

    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    frame_id = 0
    results = []
    fps = 0
    run_red_light = []

    # create filter class
    filter_class = [2, 3]

    while True:
        if frame_id % 10 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps)".format(frame_id,  fps))
        _, im = cap.read()

        if im is None:
            break

        original_im = im.copy()

        img_detect = get_area_detect(im)
        traffic_color = check_traffic_light(im)
        outputs, img_info = predictor.detect(img_detect)
        # result_frame,info = predictor.visual(outputs[0], img_info)

        if outputs[0] is not None:

            online_targets = tracker.update(
                outputs[0], [img_info['height'], img_info['width']], exp.test_size, filter_class)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    if traffic_color == "red":
                        if tlwh[1] <= 500:
                            run_red_light.append(tid)
                            original_im = cv2.line(original_im, (0, 500), (2560, 500), (0, 255, 0), thickness=2)
                            
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            online_im = plot_tracking(
                original_im, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=fps, traffic_color=traffic_color, run_red_light=len(set(run_red_light)))
            # cv2.line(online_im, (0, 500), (2560, 500), (0, 255, 0), thickness=2)

            box_light = [602.3,   0, 620.52,  50.92]
            x0 = int(box_light[0]/0.25)
            y0 = int(box_light[1]/0.25)
            x1 = int(box_light[2]/0.25)
            y1 = int(box_light[3]/0.25)

            cv2.rectangle(online_im, (x0, y0), (x1, y1), (255, 0, 0), 2)
        else:
            online_im = original_im

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        vid_writer.write(online_im)
        frame_id += 1
    # with open(res_file, 'w') as f:
    #     f.writelines(results)
    # logger.info(f"save results to {res_file}")

    cap.release()


def main(exp, args):
    predictor = Predictor(model=args.name)
    if args.demo == "image":
        image_demo(predictor, exp, args)
    elif args.demo == "video" or args.demo == "webcam":
        video_demo(predictor, exp, args)


if __name__ == '__main__':
    args = make_parser().parse_args()
    exp = get_exp_by_name(args.name)
    main(exp, args)
