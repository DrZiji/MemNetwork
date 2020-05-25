import torch
import cv2
import os
import numpy as np
import pickle
import hashlib
from torch.utils.data.dataset import Dataset

from global_config import config, modekey

class ImagnetVIDDataset(Dataset):
    def __init__(self, video_names, data_dir, z_transforms, x_transforms, db=None, training=True):
        """
        :param training:需要额外控制getitem中产生数据的流程
        """
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data_memnet.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]:x[1] for x in self.meta_data}#x[0]:video name; x[1]:trajs(该视频每一帧有几个前景目标)

        self.use_lmdb = True if db else False
        self.txn = db.begin(write=False) if self.use_lmdb else None

        # len(self.video_names)
        self.num = len(self.video_names) if config.num_per_epoch is None or not training\
                else config.num_per_epoch
        self.mode = modekey.train if training else modekey.evaluate

    def imread(self, path):
        if not self.use_lmdb:
            raise  NotImplementedError('database imread function')
        else:
            key = hashlib.md5(path.encode()).digest()
            img_buffer = self.txn.get(key)
            img_buffer = np.frombuffer(img_buffer, np.uint8)
            img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            return img


    def __getitem__(self, idx):
        """
        :return: 
            exemplar, instance: torch.Tensor [video_length, 3, ?, ?] train或eval的一个数据
            exm_bbox, ins_bbox: torch.Tensor [video_length, 4] 一个数据对应的目标框位置,左上角横纵坐标,宽高
        """
        idx = idx % len(self.video_names)
        video = self.video_names[idx]
        trajs = self.meta_data[video]
        # sample one traj
        trkid = np.random.choice(list(trajs.keys()))
        traj = trajs[trkid]#包含相同目标的一系列帧
        assert len(traj) >= config.sequence_length, "video_name: {}".format(video)
        exemplars = []
        instances = []
        exemplar_bboxs = []
        instance_bboxs = []
        if self.mode == modekey.train:
            first_idx = np.random.choice(list(range(len(traj) - config.sequence_length)))
            #取config.sequence_length长度的样本id, first_idx+1 -- len(traj)
            idx = np.arange(first_idx + 1, len(traj))
            np.random.shuffle(idx)

            idxs = idx[0:config.sequence_train].tolist()
            idxs.append(first_idx)
            idxs = sorted(idxs)
            for i in range(config.sequence_train + 1):
                img_name = os.path.join(self.data_dir, video, traj[idxs[i]][0] + ".{:02d}.x.jpg".format(trkid))
                if self.use_lmdb:
                    img = self.imread(img_name)
                else:
                    img = cv2.imread(img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                bbox = traj[idxs[i]][1]

                # visual_img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                #                              (int(bbox[0] + bbox[2]),
                #                               int(bbox[1] + bbox[3])), (255, 0, 0), 1)
                # cv2.imshow("visual_img", visual_img)
                # cv2.waitKey()

                if i > 0:#应用x_transforms
                    instance_img, instance_bbox = self.x_transforms((img, bbox))
                    instances.append(instance_img[np.newaxis, :, :, :])
                    instance_bboxs.append(instance_bbox[np.newaxis, :])

                    # debug_i_img = cv2.rectangle(instance_img, (int(instance_bbox[0]), int(instance_bbox[1])),
                    #               (int(instance_bbox[0]+instance_bbox[2]), int(instance_bbox[1]+instance_bbox[3])), (0, 0, 0), 1)
                    # cv2.imshow("debug_i_img", debug_i_img)
                    # cv2.waitKey()

                if i < config.sequence_train:#应用z_transforms
                    exemplar_img, exemplar_bbox = self.z_transforms((img, bbox))
                    exemplars.append(exemplar_img[np.newaxis, :, :, :])
                    exemplar_bboxs.append(exemplar_bbox[np.newaxis, :])

                    # debug_e_img = cv2.rectangle(exemplar_img, (int(exemplar_bbox[0]), int(exemplar_bbox[1])),
                    #             (int(exemplar_bbox[0]+exemplar_bbox[2]), int(exemplar_bbox[1]+exemplar_bbox[3])), (0, 0, 0), 1)
                    # cv2.imshow("debug_e_img", debug_e_img)
                    # cv2.waitKey()

        elif self.mode == modekey.evaluate:
            idxs = np.arange(0., float(len(traj)), float(len(traj)) / (config.sequence_eval + 1)).astype(int).tolist()
            for i in range(config.sequence_eval + 1):
                img_name = os.path.join(self.data_dir, video, traj[idxs[i]][0] + ".{:02d}.x.jpg".format(trkid))
                if self.use_lmdb:
                    img = self.imread(img_name)
                else:
                    img = cv2.imread(img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                bbox = traj[idxs[i]][1]

                if i > 0:  # 应用x_transforms
                    instance_img, instance_bbox = self.x_transforms((img, bbox))
                    instances.append(instance_img[np.newaxis, :, :, :])
                    instance_bboxs.append(instance_bbox[np.newaxis, :])

                if i < config.sequence_eval:  # 应用z_transforms
                    exemplar_img, exemplar_bbox = self.z_transforms((img, bbox))
                    exemplars.append(exemplar_img[np.newaxis, :, :, :])
                    exemplar_bboxs.append(exemplar_bbox[np.newaxis, :])
        else:
            return RuntimeError('Dataset is not applied to modekey.test')
        # 将exemplar和instance list合并成一个tensor (video_length, 3, ?, ?),第一维合并
        exemplar = torch.cat(exemplars, 0)
        exm_bbox = torch.cat(exemplar_bboxs, 0)
        instance = torch.cat(instances, 0)
        ins_bbox = torch.cat(instance_bboxs, 0)

        return exemplar, instance, exm_bbox, ins_bbox

    def __len__(self):
        return self.num

