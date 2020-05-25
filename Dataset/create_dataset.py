import numpy as np
import pickle
import os
import cv2
import functools
import xml.etree.ElementTree as ET
import sys
sys.path.append(os.getcwd())

from multiprocessing import Pool
from fire import Fire
from tqdm import tqdm
from glob import glob

from SiamFC import get_instance_image
from global_config import config

#做了以下几件事：
#1.在每一个video的子文件夹下建立相应的结构，这是生成meta_data.pkl的必要信息
#2.生成每一张图片对应的instance_image,并写入本地
def worker(output_dir, video_dir):
    # 调用一次worker方法处理一个视频
    # 在video_dir中可以把video_name提取出来，根据该video_name存储相应的pickle文件
    image_names = glob(os.path.join(video_dir, '*.JPEG'))
    image_names = sorted(image_names,
                        key=lambda x:int(x.split('/')[-1].split('.')[0]))
    video_name = video_dir.split('/')[-1]
    save_folder = os.path.join(output_dir, video_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    trajs = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        anno_name = image_name.replace('Data', 'Annotations')
        anno_name = anno_name.replace('JPEG', 'xml')
        tree = ET.parse(anno_name)
        root = tree.getroot()
        bboxes = []
        filename = root.find('filename').text
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))
            trkid = int(obj.find('trackid').text)

            instance_img, _, _, bbox_alter = get_instance_image(img, bbox,
                    config.exemplar_size, config.data_size, config.context_amount, img_mean)

            # instance_img = cv2.rectangle(instance_img, (int(bbox_alter[0]), int(bbox_alter[1])),
            #               (int(bbox_alter[0]+bbox_alter[2]), int(bbox_alter[1]+bbox_alter[3])), (0, 0, 0), 1)
            # cv2.imshow("debug_img", instance_img)
            # cv2.waitKey()

            instance_img_name = os.path.join(save_folder, filename+".{:02d}.x.jpg".format(trkid))
            cv2.imwrite(instance_img_name, instance_img) # 这里已经将crop后的图片写入本地,crop的图片是(272,272,3)大小的

            # read_img = cv2.imread(instance_img_name)
            # read_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)
            # cv2.imshow('read_img', read_img)
            # cv2.waitKey()

            if trkid in trajs:
                trajs[trkid].append((filename, bbox_alter))
            else:
                trajs[trkid] = [(filename, bbox_alter)] # 大概是指该图片中包含的前景种类
    '''
    在使用数据集时需要用到这里返回的video_name和trajs,要保证每一个video_name对应的trajs中都包含视频。
    因此将判断部分移到这里
    '''
    for trkid in list(trajs.keys()):
        if len(trajs[trkid]) <= config.sequence_length:
            del trajs[trkid]

    if trajs:
        return video_name, trajs
    else:
        return None

def processing(data_dir, output_dir, num_threads=32):
    # get all 4417 videos
    video_dir = os.path.join(data_dir, 'Data/VID')
    all_videos = glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
                 glob(os.path.join(video_dir, 'val/*'))
    # all_videos返回一个包含目标文件名的List
    meta_data = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(
            functools.partial(worker, output_dir), all_videos), total=len(all_videos)):
            if ret:
                meta_data.append(ret)
    """
    for video in all_videos:
        ret = worker(output_dir, video)
        if ret is not None:
            meta_data.append(ret)
    """
    # save meta data
    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data_memnet.pkl"), 'wb'))


if __name__ == '__main__':
    processing(output_dir='/home/s01/gdh/ILSVRC_VID_CURATION',
               data_dir='/home/s01/gdh/ILSVRC2015')

