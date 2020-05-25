import torch
import torch.nn as nn
import torch.nn.functional as F
from MemoryNetwork.base import MemNetBase
from MemoryNetwork.memnet_config import OGEMConfig


class OGEMem(MemNetBase):
    """
    :param args: tuple (embed_input_channel, embed_output_channel, gpu_id)

    移植简述:
    write: object tracking每一条测试视频只包含一个前景类;论文中特征写入memory的两条要求需要稍微修改;
        first中的threshold以跟踪结果和template correlation值作为判据;second中不考虑dissimilar情况,因为只有一个前景类,similar情况不变;
        写入object-irrelavant feature操作不考虑。

    一些符号的含义:
        Trk -- tracking result, Trk(0) = template patch
        s(i) -- search ith frame feature
        s~ -- enhanced feature
        m(i) -- memory ith content
                |           |           |           |
                |   write   |   read    |   track   |
    ------------|-----------|-----------|-----------|
    fr 1 -- n   |m(0)->m(1) |s(1)->s(1)~|           |
                |  (Trk(0)) | (att(1))  |           |
                |

    备注:
        1.memory的大小随着目标的size变化
        2.跟踪第一帧时memory就是template的reshape
    """

    def __init__(self, *args):

        super(OGEMem, self).__init__()
        embed_i_c, embed_o_c, gpu_id = args
        self.mem_embedding = [self.linear(OGEMConfig.memory_depth, embed_o_c // OGEMConfig.multi_embed) \
                                for _ in range(OGEMConfig.multi_embed)]
        self.fea_embedding = [self.linear(embed_i_c, embed_o_c // OGEMConfig.multi_embed) \
                                for _ in range(OGEMConfig.multi_embed)]
        self.mem_r_embedding = [self.linear(OGEMConfig.memory_depth, embed_o_c // OGEMConfig.multi_embed) \
                                for _ in range(OGEMConfig.multi_embed)]
        
        self.gpu_id = gpu_id
        if gpu_id >= 0:
            self.device = torch.device("cuda:" + str(gpu_id))
            with torch.cuda.device(gpu_id):
                for index in range(OGEMConfig.multi_embed):
                    self.mem_embedding[index].cuda()
                    self.fea_embedding[index].cuda()
                    self.mem_r_embedding[index].cuda()
        else:
            self.device = torch.device("cpu")

    def on_generate_write_weight(self, prev_trk, prev_f_obj_idx, time_step, batch_index, fea_len):
        """
        :param prev_trk: float; 样本的跟踪结果值(correlation后的similarity),
        :param prev_f_obj_idx: tuple[int], 上一time_step跟踪结果框内所有坐标的一维表示。
        :param time_step: 当前跟踪的时间步
        :param batch_index: int, 一个batch中样本序号, 0 -- batch-1
        :param fea_len: int, eq.7 objec b, fea_w*fea_h
        """
        assert time_step > 0, "wrong time_step, integer greater than 1 required"
        if time_step == 1:
            return True
        else:
            # att_mat_m, att_mat_sum = self.on_generate_attention(prev_obj_flatten, batch_index)
            # eq.8, off-the-shelf attention
            # att_sum = torch.sum(att_mat_sum, dim=(1, 2)) / prev_obj_flatten.shape[1]
            #获取可能要覆盖的memory中的obj的索引
            mem_idx_begin = self.mem_obj_shapes[batch_index]['fore'][0]
            mem_idx_end = self.mem_obj_shapes[batch_index]['fore'][1]

            temp_att = self.att_mat_sums[batch_index][0, mem_idx_begin: mem_idx_end, prev_f_obj_idx]
            att_sum = torch.sum(temp_att, dim=(0, 1)) / len(prev_f_obj_idx)
            temp = OGEMConfig.alpha if OGEMConfig.alpha > self.memory[batch_index][0].shape[1] / float(fea_len) \
                                        else self.memory[batch_index][0].shape[1] / float(fea_len)
            # temp = torch.tensor(temp, dtype=torch.float, device=self.device)
            # return torch.gt(att_sum, temp)
            # return True
            if att_sum.cpu().item() > temp and prev_trk > OGEMConfig.taw:
                return True
            else:
                return False

    def on_memory_write(self, prev_obj_flatten, time_step, batch_index):
        """
        :param prev_obj_flatten:torch.Tensor(),前一帧的跟踪结果;按照跟踪框范围圈定。shape = [1, w_po*h_po, c]
        :param time_step: int
        若on_generate_write_weight 返回true,则写入memory (完全覆盖)。
        """
        if time_step == 1:
            if batch_index == 0:
                self.memory = [[prev_obj_flatten]]
                self.mem_obj_shapes = [{'fore':(0, prev_obj_flatten.shape[1])}]
            else:
                self.memory.append([prev_obj_flatten])
                self.mem_obj_shapes.append({'fore':(0, prev_obj_flatten.shape[1])})
        else:
            self.memory[batch_index][0] = prev_obj_flatten # 覆盖式写入
            self.mem_obj_shapes[batch_index]['fore'] = (0, prev_obj_flatten.shape[1])

    def on_generate_read_weight(self, pres_fea_flatten, batch_index):
        """
        :param pres_fea_flatten: torch.Tensor, 当前跟踪帧的search_patch, shape = [1, w_f,*h_f, c]
        """
        b, l_f, _ = pres_fea_flatten.shape
        # att_mat_m, att_mat_sum = self.on_generate_attention(pres_fea_flatten, batch_index)
        att_mem_sum = torch.sum(self.att_mat_sums[batch_index], dim=1)
          
        auxi = torch.ones([b, l_f], dtype=torch.float, device=self.device)
        theta_index = torch.gt(att_mem_sum, auxi)
        theta_mat = 1 / torch.where(theta_index, att_mem_sum, auxi).unsqueeze(-1) # theta_mat [1, l_f] --> [1, l_f, 1]
        beta_index = torch.ge(att_mem_sum, auxi)
        beta_mat = 1 / (torch.where(beta_index, att_mem_sum, auxi) + auxi).unsqueeze(-1) # beta_mat [1, l_f] --> [1, l_f, 1]
        return theta_mat, beta_mat

    def on_memory_read(self, theta_mat, batch_index):
        """
        :param theta_mat: torch.Tensor, eq.5, shape [1, l_f, 1]
        """
        # eq.6
        mem_temp1 = [embedding(self.memory[batch_index][0]) for embedding in self.mem_r_embedding]
        mem_temp2 = [self.att_mat_ms[batch_index][0, :, :, i].unsqueeze(-1) * mem_temp1[i].unsqueeze(2) \
                        for i in range(len(mem_temp1))] # [1, l_m, l_f, c//n] list
        mem = [torch.sum(temp, dim=1) for temp in mem_temp2] # [1, l_f, c//n] list
        read = torch.cat(mem, dim=2) # [1, l_f, c]
        return read


    def on_generate_attention(self, feature_flatten, time_step, batch_index):
        """
        :param feature_flatten: torch.Tensor, shape = [1, l, c], l = w * h
        self.memory[0] shape = [b, l', c]
        由于不同的训练样本构建的memory size不同，所以不能多个batch一起计算
        """
        # 计算 feature_flatten 和 self.memory[0] 的每一行的相似性
        # feature_flatter:xi, memory:mj. similarities should be a matrix
        b, l_f, c_f = feature_flatten.shape
        _, l_m, c_m = self.memory[batch_index][0].shape

        mem_expand = self.memory[batch_index][0].unsqueeze(2)  # [1, l_m, 1, c]
        mem_expand = mem_expand.repeat([1, 1, l_f, 1])
        mem_expand = mem_expand.view(b, -1, c_m)
        fea_expand = feature_flatten.unsqueeze(1)  # [1, 1, l_f c]
        fea_expand = fea_expand.repeat([1, l_m, 1, 1])
        fea_expand = fea_expand.view(b, -1, c_f)

        cos_simi_mats = []
        att_mats = []
        for i in range(OGEMConfig.multi_embed):
            # multi head embedding
            mem_expand_embed = self.mem_embedding[i](mem_expand).view(b*l_f*l_m, -1)
            fea_expand_embed = self.fea_embedding[i](fea_expand).view(b*l_f*l_m, -1)

            # 每一行代表memory的一个row与整个feature的相似性;
            cos_simi_mat = F.cosine_similarity(mem_expand_embed, fea_expand_embed).view(b, l_m, l_f)
            # 对行向量做softmax
            att_mat = F.softmax(cos_simi_mat, dim=2)

            cos_simi_mats.append(cos_simi_mat.unsqueeze(-1))
            att_mats.append(att_mat.unsqueeze(-1))

        att_mat_m = torch.cat(att_mats, dim=-1) # [1, l_m, l_f, OGEMConfig.multi_head]
        att_mat_sum = torch.sum(att_mat_m, dim=-1) / OGEMConfig.multi_embed # [1, l_m, l_f]
        
        if time_step == 1:
            if batch_index == 0:
                self.att_mat_ms = [att_mat_m]
                self.att_mat_sums = [att_mat_sum]
            else:
                self.att_mat_ms.append(att_mat_m)
                self.att_mat_sums.append(att_mat_sum)
        else:
            self.att_mat_ms[batch_index] = att_mat_m
            self.att_mat_sums[batch_index] = att_mat_sum

    def linear(self, in_fea_c, out_fea_c):
        return nn.Linear(in_fea_c, out_fea_c)

    def forward(self, prev_fea, prev_bboxes, pres_fea, time_step, prev_trks):
        """
        :param prev_fea: torch.Tensor,包含上一帧目标的特征. shape [b, c, w_?, h_?]
        :param pres_fea: torch.Tensor,当前帧的图像特征. shape [b, c, w_s, h_s],在做Memory操作时要permute()
        :param prev_bboxes: ndarray, shape [b, 4],上一帧的目标框坐标. elements: (left_up_x, left_up_y, right_down_x, right_down_y)
        :param prev_trks: list, length: b, elements: tracking correlation score
        由于写入memory需要分开batch操作,因此整个memory step最好也分成不同batch操作。
        备注: prev_fea和prev_bboxes是配套的,在train/eval中,prev_fea 是上一帧的template feature, shape = [b, c, 6, 6]. 
            e.g. 若backbone使用AlexNet,则特征图的每一个点代表了原图中87*87的图像块,stride为8.根据object在原图中的位置,可
            以得到计算公式: 
        """
        b, c, w_s, h_s = pres_fea.shape
        pres_fea_flatten = pres_fea.view(b, c, -1).permute(0, 2, 1).contiguous()
        pres_fea_flatten = pres_fea_flatten.unsqueeze(1)

        # 测试代码
        # _, w_o, h_o, c = prev_obj.shape
        # prev_obj_flatten = prev_obj.view(b, -1, c).unsqueeze(1)
        # prev_f_obj_idxs = []
        
        out_feas = [] # 增强后的特征
        for i in range(b):
            # **********prev_fea 中裁剪prev_obj,给出obj的一维坐标表示***********
            prev_bbox = prev_bboxes[i]
            prev_obj = prev_fea[i, :, prev_bbox[0]:prev_bbox[2]+1, prev_bbox[1]:prev_bbox[3]+1]
            prev_obj_flatten = prev_obj.contiguous().view(c, -1).transpose(0, 1).unsqueeze(0)
            prev_obj_idx = []
            for j in range(prev_bbox[1], prev_bbox[3]+1):
                rela_start = j * w_s + prev_bbox[0]
                rela_end = j * w_s + prev_bbox[2] + 1
                prev_obj_idx.extend(list(range(rela_start, rela_end)))
            # ****************************************************************
            write_sig = self.on_generate_write_weight(prev_trks[i], prev_obj_idx, time_step, i, w_s*h_s) # prev_obj和m(t-1)做attention
            
            if write_sig:
                self.on_memory_write(prev_obj_flatten, time_step, i)

            self.on_generate_attention(pres_fea_flatten[i], time_step, i)

            theta_mat, beta_mat = self.on_generate_read_weight(pres_fea_flatten[i], i) # pres_fea和m(t)做attention
            mem_r = self.on_memory_read(theta_mat, i) # [1, l_f, c]
            output = beta_mat * pres_fea_flatten[i] + (1 - beta_mat) * mem_r
            out_feas.append(output)

        out_fea = torch.cat(out_feas, dim=0).view(b, w_s, h_s, c).permute(0, 3, 1, 2).contiguous()
        return out_fea
