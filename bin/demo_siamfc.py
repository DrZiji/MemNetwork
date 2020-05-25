import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import sys
sys.path.append(os.getcwd())

from fire import Fire
from bin.tracker import SiamOGEMTracker

def main(video_dir, gpu_id,  model_path):
    # load videos
    filenames = sorted(glob.glob(os.path.join(video_dir, "img/*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    gt_bboxes = pd.read_csv(os.path.join(video_dir, "groundtruth_rect.txt"), sep='\t|,| ',
            header=None, names=['xmin', 'ymin', 'width', 'height'],
            engine='python')

    title = video_dir.split('/')[-1]
    # starting tracking
    tracker = SiamOGEMTracker(model_path, gpu_id)

    prev_bbox = None
    prev_max_trk = None
    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = gt_bboxes.iloc[0].values
            tracker.init(frame, bbox)
            bbox = (bbox[0]-1, bbox[1]-1,
                    bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
        elif idx == 1: 
            bbox, max_trk = tracker.update(frame, idx)
            prev_bbox = bbox
            prev_max_trk = max_trk
        else:
            bbox, max_trk = tracker.update(frame, idx, frames[idx-1], np.array(prev_bbox), prev_max_trk)
            prev_bbox = bbox
            prev_max_trk = max_trk
        # bbox xmin ymin xmax ymax
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0),
                              2)
        gt_bbox = gt_bboxes.iloc[idx].values
        gt_bbox = (gt_bbox[0], gt_bbox[1],
                   gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3])
        frame = cv2.rectangle(frame,
                              (int(gt_bbox[0]-1), int(gt_bbox[1]-1)), # 0-index
                              (int(gt_bbox[2]-1), int(gt_bbox[3]-1)),
                              (255, 0, 0),
                              1)
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        cv2.imshow(title, frame)
        cv2.waitKey(30)

if __name__ == "__main__":
    video_dir = '/home/s03/gdh/sequences'
    video_name = os.listdir(video_dir)
    # model_path = '/home/s01/gdh/SiamFC-PyTorch/models/siamfc_kpm20.pth'
    gpu_id = 0
    for epoch in range(1, 2):
        model_name = 'siamfc_' + str(epoch)

        model_path = '/home/s03/gdh/MemNetwork/trained_models/SiamOGEM_' + str(epoch) + '.pth'
        for name in video_name:
            print('sequence', name)
            result = main(video_dir + '/' + name, gpu_id, model_path)
            if not os.path.exists('../result/' + model_name):
                os.mkdir('../result/' + model_name)
            with open('../result/' + model_name + '/' + name + '.txt', 'w', encoding='utf-8') as f:
                for bbox in result:
                    f.write(str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2] - bbox[0]) + " " + str(
                        bbox[3] - bbox[1]) + '\n')