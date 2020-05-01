import lmdb
import cv2
import numpy as np
import os 
import hashlib
import functools

from glob import glob
from fire import Fire
from tqdm import tqdm
from multiprocessing import Pool

#根据在本地的instance_image生成lmdb数据集
def worker(video_name):
    image_names = glob(video_name+'/*')
    kv = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        _, img_encode = cv2.imencode('.jpg', img)
        img_encode = img_encode.tobytes()
        kv[hashlib.md5(image_name.encode()).digest()] = img_encode
    return kv

def create_lmdb(data_dir, output_dir, num_threads):
    video_names = glob(data_dir+'/*')
    video_names = [x for x in video_names if os.path.isdir(x)]
    db = lmdb.open(output_dir, map_size=int(50e9))
    with Pool(processes=num_threads) as pool:
        
        #解释：pool.image_unordered(function, functionList)是利用多线程库调用函数function
        #在这里function用functools.partial(worker)代替。functool.partial(function, fixedFunctionParameter)
        #将函数中一部分参数固定下来，在调用时只需要传入需要频繁更改的参数。
        #关于functool.partial的更清晰的例子，在create_dataset.py中出现
        for ret in tqdm(pool.imap_unordered(
            functools.partial(worker), video_names), total=len(video_names)):
            with db.begin(write=True) as txn:
                for k, v in ret.items():
                    txn.put(k, v)

if __name__ == '__main__':
    Fire(create_lmdb)

