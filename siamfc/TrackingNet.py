import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from MemoryNetwork import OGEMem
from NetworkStructure import AlexNet
from torch import nn

from global_config import config, modekey

class SiamOGEM(nn.Module):
    def __init__(self, gpu_id, mode):
        super(SiamOGEM, self).__init__()
        # 将提特征的过程分成两部分,第一部分是提取到需要使用memnet增强特征得部分,第二部分是剩下的部分
        # 第一部分可以将batch, video_length融合提特征,第二部分在for i in range(len(pairs)):中提取
        self.features = AlexNet()
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Conv2d(384, 256, 3, 1, groups=2)

        self.memnet = OGEMem(256, 256, gpu_id)
        self.corr_bias = nn.Parameter(torch.zeros(1))
        
        self.mode = mode
        self.gpu_id = gpu_id
        if gpu_id >= 0:
            self.device = torch.device("cuda:" + str(gpu_id))
        else:
            self.device = torch.device("cpu")

        self.exemplar = None
        if self.mode != modekey.test:
            gt, weight = self._create_gt_mask((config.response_sz, config.response_sz))
            self.train_gt = torch.from_numpy(gt).to(self.device)
            self.train_weight = torch.from_numpy(weight).to(self.device)

            self.valid_gt = torch.from_numpy(gt).to(self.device)
            self.valid_weight = torch.from_numpy(weight).to(self.device)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def loss(self, pred):
        return F.binary_cross_entropy_with_logits(pred, self.gt)

    def weighted_loss(self, pred):
        if self.training:
            return F.binary_cross_entropy_with_logits(pred, self.train_gt,
                    self.train_weight, reduction='sum') / config.train_batch_size # normalize the batch_size
        else:
            return F.binary_cross_entropy_with_logits(pred, self.valid_gt,
                    self.valid_weight, reduction='sum') / config.train_batch_size # normalize the batch_size

    def bbox_ori2fea(self, bbox, cog_field, stride):
        """
        :param bbox: ndarray float [batch, 4].在图片中的目标框
        :param cog_field: int 当前特征图中一个点代表的感受野
        :param stride: 获得当前特征网络的stride
        :return ndarray int [batch, 4].在特征中的目标框
        """
        left_up_x = bbox[:, 0][:, np.newaxis]
        left_up_y = bbox[:, 1][:, np.newaxis]
        right_down_x = (bbox[:, 0] + bbox[:, 2])[:, np.newaxis]
        right_down_y = (bbox[:, 1] + bbox[:, 3])[:, np.newaxis]

        temp = np.concatenate((left_up_x, left_up_y, right_down_x, right_down_y), axis=1)
        temp = (temp - cog_field) / stride + 1
        temp = temp.astype(np.int)
        bbox_fea = np.where(temp > 0, temp, 0)
        return bbox_fea

    def _create_gt_mask(self, shape):
        # same for all pairs
        h, w = shape
        y = np.arange(h, dtype=np.float32) - (h-1) / 2.
        x = np.arange(w, dtype=np.float32) - (w-1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x**2 + y**2)
        mask = np.zeros((h, w))
        mask[dist <= config.radius / config.total_stride] = 1
        mask = mask[np.newaxis, :, :]
        weights = np.ones_like(mask)
        weights[mask == 1] = 0.5 / np.sum(mask == 1)
        weights[mask == 0] = 0.5 / np.sum(mask == 0)
        mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
        return mask.astype(np.float32), weights.astype(np.float32)

    def forward(self, *args):
        if self.mode != modekey.test:
            exemplars, instances, e_bboxs = args
            """
            exemplars/instances一定是五维tensor: [batch, videolength, c, h, w].
            e_bboxs/i_bboxs: [batch, videolength, 4]
            """
            b, vl, c, ht, wt = exemplars.shape
            _, _, _, hs, ws = instances.shape
            exemplars = exemplars.view(-1, c, ht, wt)
            instances = instances.view(-1, c, hs, ws)
            t_feas = self.features(exemplars) # t_feas = self.features(exemplars).detach_()
            s_feas = self.features(instances)

            temp, c, ht, wt = t_feas.shape
            _, _, hs, ws = t_feas.shape
            t_feas = t_feas.view(b, vl, c, ht, wt)
            s_feas = s_feas.view(b, vl, c, hs, ws)
            pairs = []

            # make feature tensor list of pairs
            self.prev_trks = [None for _ in range(b)] # 上一帧的跟踪结果最大值,初始化为None
            t_list = [torch.Tensor.view(x, [b, c, ht, wt]) for x in t_feas.split(1, dim=1)]
            s_list = [torch.Tensor.view(x, [b, c, hs, ws]) for x in s_feas.split(1, dim=1)]
            t_bbox_list = [x.view(b, 4).cpu().item() for x in e_bboxs.split(1, dim=1)]
            for i in range(vl):
                # bbox_fea 是第三层特征下的
                pairs.append(tuple(t_list[i].contiguous(), self.bbox_ori2fea(t_bbox_list[i], cog_field=43, stride=8), 
                                s_list[i].contiguous()))
            conv4 = self.conv4(pairs[0][0])
            self.exemplar = self.conv5(conv4)
            scores = []
            for i in range(len(pairs)):
                # 还没有确定prev_trks的生成方法,目前手动给定prev_trks
                prev_trks = [5. for _ in range(b)]
                # ************************************************
                pres_ins_enhanced = self.memnet(*pairs[i], i + 1, prev_trks)
                pres_ins_conv4 = self.conv4(pres_ins_enhanced)
                pres_ins_conv5 = self.conv5(pres_ins_conv4)
                score = F.conv2d(pres_ins_conv5.view([1, -1, hs, ws]), self.exemplar, groups=b)
                score = self.batch_norm(score.transpose(0, 1))
                scores.append(score)
            return torch.cat(scores, dim=1)

        else:
            params, time_step = args
            if time_step == 0:
                # 对应tracker.py中的tracker.init()
                exemplar, e_bbox = params
                conv3 = self.features(exemplar)
                conv4 = self.conv4(conv3)
                self.exemplar = self.conv5(conv4)
                self.exemplar_conv3 = conv3
                self.exemplar = torch.cat([self.exemplar for _ in range(config.num_scale)], dim=0)
                self.exemplar_conv3 = torch.cat([self.exemplar_conv3 for _ in range(config.num_scale)], dim=0)
                e_bbox = self.bbox_ori2fea(e_bbox, cog_field=43, stride=8)
                self.e_bbox = np.concatenate([e_bbox[np.newaxis, :] for _ in range(config.num_scale)])
                return None
            else:
                prev_ins, pres_ins, prev_i_bboxs, prev_trks = params
                # prev_ins/pres_ins:[scale, c, h, w]
                # prev_i_bboxs:[scale, 4]
                # prev_trk: list
                b, c, hs, ws = pres_ins.shape
                pres_ins_fea = self.features(pres_ins)
                
                assert self.exemplar, "self.exemplar should be a tensor with shape [scale, c, w_t, h_t]"

                # feed in OGEMem
                if time_step == 1:
                    prev_trks = [5 for _ in range(config.num_scale)]
                    pres_ins_enhanced = self.memnet(self.exemplar_conv3, self.e_bbox, pres_ins_fea, time_step, prev_trks)
                else:
                    prev_ins_fea_conv3 = self.features(prev_ins)
                    prev_i_bboxs = self.bbox_ori2fea(prev_i_bboxs, cog_field=43, stride=8)
                    pres_ins_enhanced = self.memnet(prev_ins_fea_conv3, prev_i_bboxs, pres_ins_fea, time_step, prev_trks)
                pres_ins_conv4 = self.conv4(pres_ins_enhanced)
                pres_ins_conv5 = self.conv5(pres_ins_conv4)
                score = F.conv2d(pres_ins_conv5.view([1, -1, hs, ws]), self.exemplar, groups=b)
                return score


