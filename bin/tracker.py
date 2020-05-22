import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import warnings
import torchvision.transforms as transforms

from torch.autograd import Variable

from SiamFC import SiamOGEM
from global_config import config, modekey
from SiamFC.custom_transforms import ToTensor
from SiamFC.utils import get_exemplar_image, get_pyramid_instance_image, get_instance_image

torch.set_num_threads(1) # otherwise pytorch will take all cpus

class SiamOGEMTracker:
    def __init__(self, model_path, gpu_id):
        self.gpu_id = gpu_id
        with torch.cuda.device(gpu_id), torch.no_grad():
            self.model = SiamOGEM(gpu_id, mode=modekey.test)
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.cuda()
            self.model.eval() 
        self.transforms = transforms.Compose([
            ToTensor()
        ])

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox, time_step=0):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
        """
        self.bbox = (bbox[0]-1, bbox[1]-1, bbox[0]-1+bbox[2], bbox[1]-1+bbox[3]) # zero based
        self.pos = np.array([bbox[0]-1+(bbox[2]-1)/2, bbox[1]-1+(bbox[3]-1)/2])  # center x, center y, zero based
        self.target_sz = np.array([bbox[2], bbox[3]])                            # width, height
        # get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z, e_bbox = get_exemplar_image(frame, self.bbox,
                config.exemplar_size, config.context_amount, self.img_mean)
        # get exemplar feature
        exemplar_img, e_bbox = self.transforms(exemplar_img, e_bbox)
        self.e_bbox = e_bbox.cpu().item()

        with torch.cuda.device(self.gpu_id), torch.no_grad():
            self.model((exemplar_img.unsqueeze(0).cuda(), self.e_bbox), time_step)

        self.penalty = np.ones((config.num_scale)) * config.scale_penalty
        self.penalty[config.num_scale//2] = 1
        # create cosine window
        self.interp_response_sz = config.response_up_stride * config.response_sz
        self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))
        # create scalse
        self.scales = config.scale_step ** np.arange(np.ceil(config.num_scale/2)-config.num_scale,
                np.floor(config.num_scale/2)+1)
        # create s_x
        self.s_x = s_z + (config.instance_size-config.exemplar_size) / scale_z

        # arbitrary scale saturation
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

    def update(self, frame, time_step, prev_frame=None, prev_i_bbox=None, prev_trks=None):
        """track object based on the previous frame
        Args:
            frame: an RGB image
        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        # 为了计算方便，拿到上一帧的跟踪结果，在上一阵中裁剪出大小为(scale, 3, 127, 127)大小的图片作为给Memnet的数据
        if prev_i_bbox and prev_frame:
            prev_instance_img = get_exemplar_image(prev_frame, prev_i_bbox, 
                    config.exemplar_size, config.context_amount, self.img_mean)
            prev_instance_img, prev_i_bbox = self.transforms(prev_instance_img, prev_i_bbox)
            prev_instance_imgs = torch.cat([prev_instance_imgs for _ in range(config.num_scale)], dim=0)
            prev_i_bboxs = torch.cat([prev_i_bboxs for _ in range(config.num_scale)], dim=0)
            with torch.cuda.device(self.gpu_id), torch.no_grad():
                prev_instance_imgs = prev_instance_imgs.cuda()
                prev_i_bboxs = prev_i_bboxs.cuda()
        else:
            prev_instance_imgs = None
            prev_i_bboxs = None

        size_x_scales = self.s_x * self.scales
        pyramid = get_pyramid_instance_image(frame, self.pos, config.instance_size, size_x_scales, self.img_mean)
        
        temp_bbox = np.array([np.NaN, np.NaN])
        instance_imgs = []
        for x in pyramid:
            instance_img, _ = self.transforms(x, temp_bbox)
            instance_imgs.append(instance_img.unsqueeze(0))
        instance_imgs = torch.cat(instance_imgs, dim=0)

        with torch.cuda.device(self.gpu_id), torch.no_grad():
            instance_imgs = instance_imgs.cuda()
            response_maps = self.model((prev_instance_imgs, instance_imgs, prev_i_bboxs, prev_trks), time_step)
            response_maps = response_maps.cpu().numpy().squeeze()
            response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC)
             for x in response_maps]
        # get max score
        max_score = np.array([x.max() for x in response_maps_up]) * self.penalty

        # penalty scale change
        scale_idx = max_score.argmax()
        scale = self.scales[scale_idx]
        response_map = response_maps_up[scale_idx]
        response_map -= response_map.min()
        response_map /= response_map.sum()
        response_map = (1 - config.window_influence) * response_map + \
                config.window_influence * self.cosine_window
        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape) # heat_map最大值的索引
        # displacement in interpolation response
        disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz-1) / 2.
        # displacement in input
        disp_response_input = disp_response_interp * config.total_stride / config.response_up_stride
        # displacement in frame
        disp_response_frame = disp_response_input * (self.s_x * scale) / config.instance_size

        # position in frame coordinates
        self.pos += disp_response_frame
        # scale damping and saturation
        self.s_x *= ((1 - config.scale_lr) + config.scale_lr * scale)
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - config.scale_lr) + config.scale_lr * scale) * self.target_sz
        bbox = (self.pos[0] - self.target_sz[0]/2 + 1, # xmin   convert to 1-based
                self.pos[1] - self.target_sz[1]/2 + 1, # ymin
                self.pos[0] + self.target_sz[0]/2 + 1, # xmax
                self.pos[1] + self.target_sz[1]/2 + 1) # ymax
        return bbox, [max_score for _ in config.num_scale]
