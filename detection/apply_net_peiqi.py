"""
Probabilistic Detectron Inference Script
"""
import core
import json
import os
from sklearn import covariance
import sys
import torch
import tqdm
from shutil import copyfile
import numpy as np
from PIL import Image
import detectron2.data.transforms as T
from vis_peiqi import final_draw
# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))
import time
# Detectron imports
from detectron2.engine import launch
from detectron2.data import build_detection_test_loader, MetadataCatalog
import cv2
# Project imports
from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from core.setup import setup_config, setup_arg_parser
from evaluator import compute_average_precision
from inference.inference_utils import instances_to_json, get_inference_output_dir, build_predictor
from detectron2.structures import BoxMode, Instances, RotatedBoxes, Boxes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from detectron2.data.detection_utils import read_image

def extract_ID(outputs):
    inst_ID = Instances((outputs.image_size[0], outputs.image_size[1]))
    pred_classes = outputs.pred_classes
    keepindex = torch.where(pred_classes!=81)

    inst_ID.pred_boxes = outputs.pred_boxes[keepindex]
    inst_ID.scores = outputs.scores[keepindex]
    inst_ID.pred_classes = outputs.pred_classes[keepindex]
    inst_ID.pred_cls_probs = outputs.pred_cls_probs[keepindex]
    inst_ID.inter_feat = outputs.inter_feat[keepindex]
    inst_ID.det_labels = outputs.det_labels[keepindex]
    inst_ID.pred_boxes_covariance = outputs.pred_boxes_covariance[keepindex]
    inst_ID.complete_scores = outputs.complete_scores[keepindex]
    return inst_ID


def main(args):
    # Setup config
    cfg = setup_config(args,
                       random_seed=args.random_seed,
                       is_testing=True)
    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    # cfg.DATALOADER.NUM_WORKERS = 32
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type

 
    train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
    test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        args.test_dataset).thing_dataset_id_to_contiguous_id

    # If both dicts are equal or if we are performing out of distribution
    # detection, just flip the test dict.
    cat_mapping_dict = get_train_contiguous_id_to_test_thing_dataset_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id)
    cat_mapping_dict.update({81:81})

    # Build predictor
    predictor = build_predictor(cfg)
    
    start = time.time()

    frame_rate = 0.5
    prev = 0
    cap = cv2.VideoCapture(0)

    while True:
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if time_elapsed > 1./frame_rate:
            prev = time.time()
            cv2.imwrite('temp_image.png', frame)
            original_image = read_image('temp_image.png')
            # Do same preprocessing as DefaultPredictor
            aug = T.ResizeShortestEdge(
                    [800, 800], 1333
                )
            image = aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            height, width = original_image.shape[:2]

            final_output_list_idood = [] # test mixed dataset, include ood, id
            if not args.eval_only:
                with torch.no_grad():
                    input_im = [{
                        'file_name':'temp_image.png',
                        'height':height,
                        'width':width,
                        'image':image
                    }]
                    outputs = predictor(input_im, args.pretest)
                    final_output_list_idood.extend(
                                instances_to_json(
                                    outputs,
                                    1,
                                    cat_mapping_dict))
            
            combined_img = final_draw(final_output_list_idood,input_im[0]['file_name'],'./output/',input_im[0]['file_name'])
            cv2.imshow("Detected Objects", combined_img)
        k = cv2.waitKey(2)
        if k != -1:
            break
    
    # ###load###
    # for i in range(50):
    #     original_image = read_image('image_6.png')
    #         # Do same preprocessing as DefaultPredictor
    #     aug = T.ResizeShortestEdge(
    #             [800, 800], 1333
    #         )
    #     image = aug.get_transform(original_image).apply_image(original_image)
    #     image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    #     height, width = original_image.shape[:2]

    #     final_output_list_idood = [] # test mixed dataset, include ood, id
    #     if not args.eval_only:
    #         with torch.no_grad():
    #             input_im = [{
    #                 'file_name':'image_6.png',
    #                 'height':height,
    #                 'width':width,
    #                 'image':image
    #             }]
    #             outputs = predictor(input_im, args.pretest)
    #             final_output_list_idood.extend(
    #                         instances_to_json(
    #                             outputs,
    #                             1,
    #                             cat_mapping_dict))
        
    #     final_draw(final_output_list_idood,input_im[0]['file_name'],'./output/',input_im[0]['file_name'])

    end = time.time()
        # with open('./test.json', 'w') as fp:
        #     json.dump(final_output_list_idood, fp, indent=4, separators=(',', ': '))




if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    # Support single gpu inference only.
    args.num_gpus = 1
    # args.num_machines = 8

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
