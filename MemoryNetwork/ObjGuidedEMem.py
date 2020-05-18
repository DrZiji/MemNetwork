import torch
import torch.nn as nn
import torch.nn.functional as F
from base import MemNetBase
from memnet_config import OGEMemConfig


class OGEMem(MemNetBase):
    """
    移植简述:
    write: object tracking每一条测试视频只包含一个前景类;论文中特征写入memory的两条要求需要稍微修改;
        first中的threshold以跟踪结果和template correlation值作为判据;second中不考虑dissimilar情况,因为只有一个前景类,similar情况不变;
        写入object-irrelavant feature操作不考虑。

    一些符号的含义:
        t -- template feature
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
        embed_i_c, embed_o_c = args
        self.mem_embedding = [self.linear(OGEMemConfig.memory_depth, embed_o_c // OGEMemConfig.multi_embed) 
                                for _ in range(OGEMemConfig.multi_embed)]
        self.fea_embedding = [self.linear(embed_i_c, embed_o_c // OGEMemConfig.multi_embed)
                                for _ in range(OGEMemConfig.multi_embed)]

    def on_generate_write_weight(self, obj_feature_flatten, time_step, att_mat_sum=None):
        """
        :param obj_feature_flatten: torch.Tensor; 一个batch中不同样本的obj_feature
        :param att_mat_sum: torch.Tensor,一个batch中的attention矩阵
        """
        if time_step == 1:
            return True
        else:
            assert att_mat_sum, "wrong att_mat_sum type, need a tensor in shape [batch, length_memory, length_obj], got None"
            # eq.8
            att_sum = torch.sum(att_mat_sum, dim=(1, 2)) / obj_feature_flatten.shape[1]
            if att_sum > torch.max()
            pass

    def on_memory_write(self, prev_obj, time_step, batch_index):
        """
        :param prev_obj:torch.Tensor(),前一帧的跟踪结果;按照跟踪框范围圈定。shape = [b, w_po, h_po, c]
        :param time_step: int
        若on_generate_write_weight 返回true,则写入memory (完全覆盖)。
        """
        b, w_po, h_po, c = prev_obj.shape
        prev_obj_flatten = prev_obj.view(b, -1, c)

        pass

    def on_generate_read_weight(self):
        pass

    def on_memory_read(self):
        pass

    def on_generate_attention(self, feature_flatten, batch_index):
        """
        :param feature_flatten: torch.Tensor, shape = [b, l, c], l = w * h
        self.memory shape = [b, l', c]
        由于不同的训练样本构建的memory size不同，所以不能多个batch一起计算
        """
        # 计算 feature_flatten 和 self.memory 的每一行的相似性
        # feature_flatter:xi, memory:mj. similarities should be a matrix
        b, l_f, c_f = feature_flatten.shape
        _, l_m, c_m = self.memory[batch_index].shape

        mem_expand = self.memory[batch_index].unsqueeze(2)  # [b, l_m, 1, c]
        mem_expand = mem_expand.repeat([1, 1, l_f, 1])
        mem_expand = mem_expand.view(b, -1, c_m)
        fea_expand = feature_flatten.unsqueeze(1)  # [b, 1, l_f c]
        fea_expand = fea_expand.repeat([1, l_m, 1, 1])
        fea_expand = fea_expand.view(b, -1, c_f)

        cos_simi_mats = []
        att_mats = []
        for i in range(OGEMemConfig.multi_embed):
            # multi head embedding
            mem_expand_embed = self.mem_embedding[i](mem_expand).view(b*l_f*l_m, -1)
            fea_expand_embed = self.fea_embedding[i](fea_expand).view(b*l_f*l_m, -1)

            # 每一行代表memory的一个row与整个feature的相似性;
            cos_simi_mat = F.cosine_similarity(mem_expand_embed, fea_expand_embed).view(b, l_m, l_f)
            # 对行向量做softmax
            att_mat = F.softmax(cos_simi_mat, dim=2)

            cos_simi_mats.append(cos_simi_mat.unsqueeze(-1))
            att_mats.append(att_mat.unsqueeze(-1))

        att_mat_m = torch.cat(att_mats, dim=-1)
        att_mat_sum = torch.sum(att_mat_m, dim=-1) / OGEMemConfig.multi_embed

        return att_mat_m, att_mat_sum

    def linear(self, in_fea_c, out_fea_c):
        return nn.Linear(in_fea_c, out_fea_c)

    def forward(self, prev_feature, prev_bboxes, pres_feature):
        """
        :param prev_feature: torch.Tensor, shape [b, w, h, c]
        :param pres_feature: torch.Tensor, shape [b, w, h, c]
        :param bboxes: list, length: b, elements: bbox(otb style)
        由于写入memory需要分开batch操作，因此整个memory step 最好也分成不同batch操作。
        """
        b, w_s, h_s, c = prev_feature.shape

        for i in range(b):
            # prev_feature 中裁剪 prev_obj
            write_boolean = self.on_generate_write_weight() # prev_obj_feature和m(t-1)做attention
            if write_boolean:
                self.on_memory_write() # 
            
            self.on_generate_read_weight() # pres_feature 和 m(t)做attention
            self.on_memory_read() #


        
