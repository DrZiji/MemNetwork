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
        T -- template feature
        s(i) -- search ith frame feature
        s~ -- enhanced feature
        m(i) -- memory ith content
                |           |           |           |
                |   write   |   read    |   track   |
    ------------|-----------|-----------|-----------|
    fr 1 -- n   |m(0)->m(1) |s(1)->s(1)~|           |
                |           |           |           |

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
        else:
            self.device = torch.device("cpu")

    def on_generate_write_weight(self, prev_obj_flatten, prev_trk, time_step, batch_index, fea_len):
        """
        :param prev_obj_flatten: torch.Tensor; 一个batch中不同样本的obj_feature，shape = [1, l_po, c]
        :param prev_trk: float; 样本的跟踪结果值(correlation后的similarity)， 
        :param time_step: 当前跟踪的时间步
        :param batch_index: int, 一个batch中样本序号, 0 -- batch-1
        :param fea_len: int, fea_w*fea_h
        """
        assert time_step > 0, "wrong time_step, integer greater than 1 required"
        if time_step == 1:
            return True
        else:
            att_mat_m, att_mat_sum = self.on_generate_attention(prev_obj_flatten, batch_index)
            # eq.8
            att_sum = torch.sum(att_mat_sum, dim=(1, 2)) / prev_obj_flatten.shape[1]
            temp = OGEMConfig.alpha if OGEMConfig.alpha > self.memory[batch_index].shape[1] / float(fea_len) \
                                        else self.memory[batch_index].shape[1] / float(fea_len)
            # temp = torch.tensor(temp, dtype=torch.float, device=self.device)
            # return torch.gt(att_sum, temp)
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
                self.memory = [prev_obj_flatten]
            else:
                self.memory.append(prev_obj_flatten)
        else:
            self.memory[batch_index] = prev_obj_flatten # 覆盖式写入

    def on_generate_read_weight(self, pres_fea_flatten, batch_index):
        """
        :param pres_fea_flatten: torch.Tensor, 当前跟踪帧的search_patch, shape = [1, w_f,*h_f, c]
        """
        b, l_f, _ = pres_fea_flatten.shape
        att_mat_m, att_mat_sum = self.on_generate_attention(pres_fea_flatten, batch_index)
        att_mem_sum = torch.sum(att_mat_sum, dim=1)
          
        auxi = torch.ones([b, l_f], dtype=torch.float, device=self.device)
        theta_index = torch.gt(att_mem_sum, auxi)
        theta_mat = 1 / torch.where(theta_index, att_mem_sum, auxi).unsqueeze(-1) # theta_mat [1, l_f] --> [1, l_f, 1]
        beta_index = torch.ge(att_mem_sum, auxi)
        beta_mat = 1 / (torch.where(beta_index, att_mem_sum, auxi) + auxi).unsqueeze(-1) # beta_mat [1, l_f] --> [1, l_f, 1]
        return theta_mat, beta_mat, att_mat_m

    def on_memory_read(self, theta_mat, att_mat_m, batch_index):
        """
        :param theta_mat: torch.Tensor, eq.5, shape [1, l_f, 1]
        :param att_mat_m: torch.Tensor, mutiple embedding head attention, shape [1, l_m, l_f, OGEMConfig.multi_embed]
        """
        # eq.6
        mem_temp1 = [embedding(self.memory[batch_index]) for embedding in self.mem_r_embedding]
        mem_temp2 = [att_mat_m[0, :, :, i].unsqueeze(-1) * mem_temp1[i].unsqueeze(2) for i in range(len(mem_temp1))] # [1, l_m, l_f, c//n] list
        mem = [torch.sum(temp, dim=1) for temp in mem_temp2] # [1, l_f, c//n] list
        read = torch.cat(mem, dim=2) # [1, l_f, c]
        return read


    def on_generate_attention(self, feature_flatten, batch_index):
        """
        :param feature_flatten: torch.Tensor, shape = [1, l, c], l = w * h
        self.memory shape = [b, l', c]
        由于不同的训练样本构建的memory size不同，所以不能多个batch一起计算
        """
        # 计算 feature_flatten 和 self.memory 的每一行的相似性
        # feature_flatter:xi, memory:mj. similarities should be a matrix
        b, l_f, c_f = feature_flatten.shape
        _, l_m, c_m = self.memory[batch_index].shape

        mem_expand = self.memory[batch_index].unsqueeze(2)  # [1, l_m, 1, c]
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

        return att_mat_m, att_mat_sum

    def linear(self, in_fea_c, out_fea_c):
        return nn.Linear(in_fea_c, out_fea_c)

    def forward(self, prev_fea, prev_bboxes, prev_trk, pres_fea, time_step, prev_obj):
        """
        :param prev_feature: torch.Tensor, shape [b, w_s, h_s, c]
        :param pres_feature: torch.Tensor, shape [b, w_s, h_s, c]
        :param prev_bboxes: list, length: b, elements: bbox(otb style)
        :param prev_trk: list, length: b, elements: tracking correlation score
        由于写入memory需要分开batch操作，因此整个memory step 最好也分成不同batch操作。
        """
        b, w_s, h_s, c = prev_fea.shape
        pres_fea_flatten = pres_fea.view(b, -1, c).unsqueeze(1)

        _, w_o, h_o, c = prev_obj.shape
        prev_obj_flatten = prev_obj.view(b, -1, c).unsqueeze(1)

        outputs = []
        for i in range(b):
            # *****prev_fea 中裁剪prev_obj,目前未实现*****
            # prev_obj = None
            # l_po = 1
            # *******************************************
            # prev_obj_flatten = prev_obj.view(1, l_po, c)

            write_sig = self.on_generate_write_weight(prev_obj_flatten[i], prev_trk[i], time_step, i, w_s*h_s) # prev_obj和m(t-1)做attention
            if write_sig:
                self.on_memory_write(prev_obj_flatten[i], time_step, i)

            theta_mat, beta_mat, att_mat_m_r = self.on_generate_read_weight(pres_fea_flatten[i], i) # pres_fea和m(t)做attention
            mem_r = self.on_memory_read(theta_mat, att_mat_m_r, i) # [1, l_f, c]
            output = beta_mat * pres_fea_flatten[i] + (1 - beta_mat) * mem_r
            outputs.append(output)

        out_fea = torch.cat(outputs, dim=0).view(b, w_s, h_s, c)
        return out_fea
