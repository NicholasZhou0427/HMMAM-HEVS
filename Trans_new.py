# #!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import math
# from torch.utils.checkpoint import checkpoint  # 已移除：与 DDP 冲突

from data.Hevrpstw_data import StateHEVRPSTW
from functions import move_to
from Sublayers import EncoderLayer
from WandbOption.MyOption import VRPOption


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        return x + self.dummy - self.dummy  # (also tried x+self.dummy)


class Trans_Model_Encoder(nn.Module):

    def __init__(self, embed_dim, n_layers, n_heads, FF_dim, opts):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.opts = opts

        # self.dummy_layer = DummyLayer()

        # 仓库点和充电桩点是否重新编码????????????????????????????????????????
        self.init_embed_d = nn.Linear(6, embed_dim, bias=True)
        self.init_embed_f = nn.Linear(6, embed_dim, bias=True)
        self.init_embed_c = nn.Linear(6, embed_dim, bias=True)
        # self.init_embed_d = nn.Linear(7, embed_dim, bias=True)
        # self.init_embed_f = nn.Linear(7, embed_dim, bias=True)
        # self.init_embed_c = nn.Linear(7, embed_dim, bias=True)
        # self.init_embed_d = nn.Linear(3, embed_dim, bias=True)
        # self.init_embed_f = nn.Linear(3, embed_dim, bias=True)
        # self.init_embed_c = nn.Linear(3, embed_dim, bias=True)
        self.init_embed = nn.Linear(embed_dim, embed_dim)

        self.init_dist = nn.Linear(opts.graph_size, embed_dim, bias=True)
        self.init_dist2 = nn.Linear(embed_dim * 2, embed_dim, bias=True)

        # 是否需要加这一层???????????
        self.dropout1 = nn.Dropout(p=opts.dropout_prob)
        # LN还是BN？？？？？？？？？？？？？？？？？
        self.LN1 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(embed_dim, n_heads, FF_dim, opts.dropout_prob)
            for _ in range(n_layers)])

        # self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def input_embedding(self, x_new):
        x_new2 = self.init_embed_c(x_new)
        x_new2[:, 0, :] = self.init_embed_d(x_new[:, 0, :])
        x_new2[:, 1:21, :] = self.init_embed_f(x_new[:, 1:21, :])
        x_new2 = self.init_embed(x_new2)
        return x_new2

    def forward(self, x, d, t_start, t_end, t_service, t_mid, mask=None):
        # 输入embedding
        # 为了加入checkpoint后输入有梯度
        # x = self.dummy_layer(input)

        # d_new用不用加？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        d_new = (d / self.opts.hc_VEHICLE_CAPACITY[0])
        for i in range(1, self.opts.VEHICLE_num):
            d_new = torch.cat((d_new, d / self.opts.hc_VEHICLE_CAPACITY[i]), -1)
        # d = d_new

        x_new = torch.cat([x, d, t_start, t_end, t_service], dim=-1)
        x_new = self.input_embedding(x_new)
        # x_new = checkpoint(self.input_embedding, x_new)

        tmp_coords = x
        tmp_dist = (tmp_coords[:, :, None, :] - tmp_coords[:, None, :, :]).norm(p=2, dim=-1)
        tmp_dist = self.init_dist(tmp_dist)
        x_new = torch.cat([x_new, tmp_dist], dim=-1)
        x_new = self.init_dist2(x_new)
        # x_new = checkpoint(self.init_dist2, x_new)

        # 是否需要这一层????????????????????????????
        x = self.LN1(self.dropout1(x_new))
        # x = self.LN1(x_new)
        # 编码器attention
        for enc_layer in self.layer_stack:
            x = enc_layer(x, src_mask=None)
            # x = checkpoint(enc_layer, x)

        # 确保变量连续
        x = x.contiguous()
        avg = torch.mean(x, dim=1)

        return x, avg

class Trans_Model_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return


class Trans_Model_Decoder_low(Trans_Model_Decoder):
    def __init__(self, embed_dim, opts):
        super(Trans_Model_Decoder, self).__init__()

        self.opts = opts
        self.decode_type = None
        self.kv_size = embed_dim // self.opts.n_heads

        # 解码器几个固定变量，两个注意力的k,v
        self.fixed_context = None
        self.glimpse_key_fixed = None
        self.glimpse_val_fixed = None
        self.logit_key_fixed = None
        self.glimpse_key_fixed_veh = None
        self.glimpse_val_fixed_veh = None
        self.logit_key_fixed_veh = None

        # 用不用加这个参数：bias=False，可以试验一下
        self.project_step_context = nn.Linear(embed_dim + 3, embed_dim, bias=False)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        # 有这行参数即使不用，效果也会变差？？？？？？？？？
        # self.project_query_context = nn.Linear(embed_dim, embed_dim, bias=False)

        self.project_node_embeddings = nn.Linear(embed_dim, 6 * embed_dim, bias=False)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_out_veh = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_logits_veh = nn.Linear(self.opts.graph_size, 1, bias=False)

        # 选车模块参数----------------------------------------------------------------------------------
        # 选车时只考虑每个车的坐标和已行驶路径长
        self.node_veh = 6
        self.FF_veh = nn.Sequential(
            nn.Linear(self.node_veh, embed_dim),
            nn.Linear(embed_dim, opts.hc_decoder_FF_dim),
            nn.ReLU(),
            nn.Linear(opts.hc_decoder_FF_dim, embed_dim)
        ) if opts.hc_decoder_FF_dim > 0 else nn.Linear(embed_dim, embed_dim)
        self.FF_tour = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, opts.hc_decoder_FF_dim),
            nn.ReLU(),
            nn.Linear(opts.hc_decoder_FF_dim, embed_dim)
        ) if opts.hc_decoder_FF_dim > 0 else nn.Linear(embed_dim, embed_dim)
        self.select_veh_embed = nn.Linear(embed_dim * 2, self.opts.VEHICLE_num)
        # 选车模块参数----------------------------------------------------------------------------------

        # self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type

    def get_fixed(self, embeddings, avg):
        batch_size, graph_size, embed_dim = embeddings.size()
        # 得到两个注意力的k,v
        self.fixed_context = self.project_fixed_context(avg)
        self.glimpse_key_fixed, self.glimpse_val_fixed, self.logit_key_fixed, \
            self.glimpse_key_fixed_veh, self.glimpse_val_fixed_veh, self.logit_key_fixed_veh \
            = self.project_node_embeddings(embeddings).chunk(6, dim=-1)
        self.glimpse_key_fixed = self.glimpse_key_fixed.view(batch_size, graph_size, self.opts.n_heads, -1) \
            .permute(2, 0, 1, 3)
        self.glimpse_val_fixed = self.glimpse_val_fixed.view(batch_size, graph_size, self.opts.n_heads, -1) \
            .permute(2, 0, 1, 3)
        self.glimpse_key_fixed_veh = self.glimpse_key_fixed_veh.view(batch_size, graph_size, self.opts.n_heads, -1) \
            .permute(2, 0, 1, 3)
        self.glimpse_val_fixed_veh = self.glimpse_val_fixed_veh.view(batch_size, graph_size, self.opts.n_heads, -1) \
            .permute(2, 0, 1, 3)

    def forward(self, embeddings, state, mask, veh):
        batch_size, graph_size, embed_dim = embeddings.size()
        # 得到context    [batch, embeddings_dim + 1]
        context_temp = self.get_context(embeddings, state, veh)
        # 得到query      [batch, 1, embeddings_dim]
        # query到底是拼接还是相加???????????????????????????????????????????????????????????
        query = self.fixed_context + self.project_step_context(context_temp)
        # 加了个全连接，性能变差了5.0669-5.1122
        # query = self.project_query_context(query)

        # 待实现的sdevrp接口
        glimpse_key = self.glimpse_key_fixed
        glimpse_val = self.glimpse_val_fixed
        logit_key = self.logit_key_fixed

        logits_node = self.get_select_p(query, glimpse_key, glimpse_val, logit_key, self.kv_size, mask)

        return logits_node, query

    def select_veh_new(self, state, embeddings, obj, outputs_sequences, path_len, mask_veh_node):
        # Route特征和车辆特征到底选哪些，参考论文再好好想想！！！！！！！！！！！！！！！！！！！！！！！！
        # 加上时间窗后，这些函数都要考虑怎么加上相关信息！！！！！！！！！！！！！！！！
        # 是否需要考虑多车容量和速度的关系？？？？？？？？？？？？？？？？？
        # 选车时也用上mask信息！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 选车时也用上mask信息！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 选车时也用上mask信息！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 选车时也用上mask信息！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        tour_dis = state.lengths  # [batch_size, num_veh]

        if obj == 'min-max':
            SPEED = [1 for _ in range(self.opts.VEHICLE_num)]
        if obj == 'min-sum':
            # 这里是否要改，看论文！！！！！！！！！！！！！！！！！！！！！！！！！
            SPEED = [1 for _ in range(self.opts.VEHICLE_num)]

        batch_size, graph_size, embed_dim = embeddings.size()

        current_loc = state.cur_coord.transpose(0, 1)  # [num_veh, batch_size, 2]
        current_time = state.time
        # 这里用的剩余容量电量
        current_used_capacity = self.opts.hc_VEHICLE_CAPACITY[0] - state.used_capacity
        current_used_Q = self.opts.VEHICLE_Q - state.used_Q
        veh_context = torch.cat(  # [batch_size, 6]
            (
                tour_dis[:, 0].unsqueeze(-1),  # [batch_size, 1]
                current_loc[0, :],  # [batch_size, 2])
                current_time[:, 0].unsqueeze(-1),
                current_used_capacity[:, 0].unsqueeze(-1),
                current_used_Q[:, 0].unsqueeze(-1),
            ),
            -1
        ).unsqueeze(1)  # [batch_size, 6]-># [batch_size, 1, 6]
        for i in range(1, self.opts.VEHICLE_num):
            veh_context_part = torch.cat(  # [batch_size, 6*VEHICLE_num]
                (
                    tour_dis[:, i].unsqueeze(-1),  # [batch_size, 1]
                    current_loc[i, :],  # [batch_size, 2])
                    current_time[:, i].unsqueeze(-1),
                    current_used_capacity[:, i].unsqueeze(-1),
                    current_used_Q[:, i].unsqueeze(-1),
                ),
                -1
            ).unsqueeze(1)
            veh_context = torch.cat((veh_context, veh_context_part), dim=1)

        if path_len == 0:
            mean_tour = torch.zeros([batch_size, self.opts.VEHICLE_num, embed_dim]).float().to(embeddings.device)
        else:
            tour_list_tmp, tour_con_list_tmp = [], []
            for i in range(self.opts.VEHICLE_num):
                tour_list_tmp.append(outputs_sequences[:, i, :path_len])
            for i in range(self.opts.VEHICLE_num):
                ys = torch.gather(
                    embeddings,  # [batch_size, graph_size, embed_dim]
                    1,
                    (tour_list_tmp[i].clone())[..., None].contiguous().expand(batch_size, tour_list_tmp[i].size(-1),
                                                                              embed_dim)
                )
                tour_con_list_tmp.append(ys)

            # 可以试试mean(试了一次效果不行，生成不了可行解)
            mean_tour = torch.max(tour_con_list_tmp[0], dim=1)[0].unsqueeze(1)
            # mean_tour = torch.mean(tour_con_list_tmp[0], dim=1).unsqueeze(1)
            for i in range(1, self.opts.VEHICLE_num):
                mean_tour_part = torch.max(tour_con_list_tmp[i], dim=1)[0].unsqueeze(1)
                # mean_tour_part = torch.mean(tour_con_list_tmp[i], dim=1).unsqueeze(1)
                mean_tour = torch.cat((mean_tour, mean_tour_part), dim=1)

        veh_context = self.FF_veh(veh_context)  # [batch_size, embed_dim]
        tour_context = self.FF_tour(mean_tour)
        # 先试试直接add
        # context = torch.cat((veh_context, tour_context), -1).view(batch_size, embed_dim * 2)
        # 再尝试加一下avg
        tmp_avg = torch.mean(embeddings, dim=1).unsqueeze(1).expand_as(veh_context)
        # context = veh_context + tour_context + tmp_avg
        context = veh_context + tour_context

        # 选车attention-----------------------------------------------------------------------------
        # tmp_k = torch.mean(embeddings, dim=1).unsqueeze(1)
        # tmp_k_veh = tmp_k.view(batch_size, 1, self.opts.n_heads, self.kv_size).permute(2, 0, 1, 3)
        # glimpse_Q_veh = context.view(batch_size, self.opts.VEHICLE_num, self.opts.n_heads, self.kv_size).permute(2, 0, 1, 3)
        # compatibility_veh = torch.matmul(glimpse_Q_veh, tmp_k_veh.transpose(-2, -1)) / math.sqrt(self.kv_size)
        # heads_veh = torch.matmul(torch.softmax(compatibility_veh, dim=-1), tmp_k_veh)
        # glimpse_veh = self.project_out_veh(heads_veh.permute(1, 2, 0, 3).contiguous().view(batch_size, self.opts.VEHICLE_num, embed_dim))
        # logits_veh = torch.matmul(glimpse_veh, tmp_k.transpose(-2, -1)).squeeze(-2) / math.sqrt(self.kv_size)
        # logits_veh = torch.tanh(logits_veh) * self.opts.tanh_clipping
        # logits_veh = logits_veh.squeeze()
        # 选车attention-----------------------------------------------------------------------------
        # 这里没用mask
        glimpse_Q_veh = context.view(batch_size, self.opts.VEHICLE_num, self.opts.n_heads, self.kv_size).permute(2, 0, 1, 3)
        compatibility_veh = torch.matmul(glimpse_Q_veh, self.glimpse_key_fixed_veh.transpose(-2, -1)) / math.sqrt(self.kv_size)
        # 尝试下mask,好像效果不太行
        # compatibility_veh [n_heads, batch_size, VEHICLE_num, graph_size]
        # mask_veh_node [batch_size, VEHICLE_num, graph_size]
        # compatibility_veh[mask_veh_node[None, :, :, :].expand_as(compatibility_veh)] = -math.inf
        heads_veh = torch.matmul(torch.softmax(compatibility_veh, dim=-1), self.glimpse_val_fixed_veh)
        glimpse_veh = self.project_out_veh(
            heads_veh.permute(1, 2, 0, 3).contiguous().view(batch_size, self.opts.VEHICLE_num, embed_dim))
        logits_veh = torch.matmul(glimpse_veh, self.logit_key_fixed_veh.transpose(-2, -1)).squeeze(-2) / math.sqrt(self.kv_size)
        # todo logits_veh = torch.tanh(logits_veh) * self.opts.tanh_clipping
        logits_veh = torch.clamp(logits_veh, -self.opts.tanh_clipping, self.opts.tanh_clipping)
        # 第2个mask还没加，先只试试第一个mask!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # logits_veh[mask_veh_node[None, :, :, :].expand_as(compatibility_veh)] = -math.inf
        logits_veh = self.project_logits_veh(logits_veh).squeeze()
        # print(logits_veh)
        # logits_veh[mask] = -math.inf
        # 选车attention-----------------------------------------------------------------------------
        # 尝试加入图信息
        # avg = torch.mean(embeddings, dim=1).squeeze()
        # context = torch.cat((context, avg), -1).view(batch_size, embed_dim * 3)

        # logits_veh = self.select_veh_embed(context)
        # 这里也加一个梯度裁剪！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 缩放到相同尺度，便于HR！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        ## logits_veh = torch.tanh(logits_veh) * self.opts.tanh_clipping
        return logits_veh, context

    def get_select_p(self, q, k1, v1, k2, kv_size, mask):
        glimpse_Q = q.view(-1, 1, self.opts.n_heads, kv_size).permute(2, 0, 1, 3)
        compatibility = torch.matmul(glimpse_Q, k1.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        compatibility[mask[None, :, None, :].expand_as(compatibility)] = -math.inf
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), v1)
        glimpse = self.project_out(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, 1, self.opts.n_heads * kv_size))
        logits = torch.matmul(glimpse, k2.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))
        # todo logits = torch.tanh(logits) * self.opts.tanh_clipping
        logits = torch.clamp(logits, -self.opts.tanh_clipping, self.opts.tanh_clipping)
        return logits

    def get_context(self, embeddings, state, veh):
        # 还有这里到底是图嵌入+前一节点+容/电量cat还是相加？？？？？？？？？？？？？？？？？？？？？？？？？
        # 到底如何改进
        current_node = state.prev_a
        batch_size, _ = current_node.size()
        # 仅用于去函数波浪线
        yyysss = self.decode_type
        return torch.cat(
            (
                torch.gather(
                    embeddings,
                    1,
                    (current_node[torch.arange(batch_size), veh]).contiguous()
                    .view(batch_size, 1, 1).expand(batch_size, 1, embeddings.size(-1))
                ).view(batch_size, embeddings.size(-1)),  # [batch_size, embed_dim]
                (state.VEHICLE_CAPACITY[None, veh].to(embeddings.device) - state.used_capacity[
                    torch.arange(batch_size), veh]).transpose(0, 1),  # [batch_size, 1]
                (state.VEHICLE_Q - state.used_Q[torch.arange(batch_size), veh])[:, None],  # [batch_size, 1]
                (state.time[torch.arange(batch_size), veh])[:, None]
            ),
            -1
        )

class Trans_Model_Decoder_high(Trans_Model_Decoder):
    def __init__(self, embed_dim, opts):
        super(Trans_Model_Decoder, self).__init__()

        self.opts = opts
        self.decode_type = None
        self.kv_size = embed_dim // self.opts.n_heads

        # 解码器几个固定变量，两个注意力的k,v
        self.fixed_context = None
        self.glimpse_key_fixed = None
        self.glimpse_val_fixed = None
        self.logit_key_fixed = None
        self.glimpse_key_fixed_veh = None
        self.glimpse_val_fixed_veh = None
        self.logit_key_fixed_veh = None

        # 用不用加这个参数：bias=False，可以试验一下
        self.project_step_context = nn.Linear(embed_dim + 3, embed_dim, bias=False)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        # 有这行参数即使不用，效果也会变差？？？？？？？？？
        # self.project_query_context = nn.Linear(embed_dim, embed_dim, bias=False)

        self.project_node_embeddings = nn.Linear(embed_dim, 6 * embed_dim, bias=False)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_out_veh = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_logits_veh = nn.Linear(self.opts.graph_size, 1, bias=False)

        # 选车模块参数----------------------------------------------------------------------------------
        # 选车时只考虑每个车的坐标和已行驶路径长
        self.node_veh = 6
        self.FF_veh = nn.Sequential(
            nn.Linear(self.node_veh, embed_dim),
            nn.Linear(embed_dim, opts.hc_decoder_FF_dim),
            nn.ReLU(),
            nn.Linear(opts.hc_decoder_FF_dim, embed_dim)
        ) if opts.hc_decoder_FF_dim > 0 else nn.Linear(embed_dim, embed_dim)
        self.FF_tour = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, opts.hc_decoder_FF_dim),
            nn.ReLU(),
            nn.Linear(opts.hc_decoder_FF_dim, embed_dim)
        ) if opts.hc_decoder_FF_dim > 0 else nn.Linear(embed_dim, embed_dim)
        self.select_veh_embed = nn.Linear(embed_dim * 2, self.opts.VEHICLE_num)
        # 选车模块参数----------------------------------------------------------------------------------

        # self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type

    def get_fixed(self, embeddings, avg):
        batch_size, graph_size, embed_dim = embeddings.size()
        # 得到两个注意力的k,v
        self.fixed_context = self.project_fixed_context(avg)
        self.glimpse_key_fixed, self.glimpse_val_fixed, self.logit_key_fixed, \
            self.glimpse_key_fixed_veh, self.glimpse_val_fixed_veh, self.logit_key_fixed_veh \
            = self.project_node_embeddings(embeddings).chunk(6, dim=-1)
        self.glimpse_key_fixed = self.glimpse_key_fixed.view(batch_size, graph_size, self.opts.n_heads, -1) \
            .permute(2, 0, 1, 3)
        self.glimpse_val_fixed = self.glimpse_val_fixed.view(batch_size, graph_size, self.opts.n_heads, -1) \
            .permute(2, 0, 1, 3)
        self.glimpse_key_fixed_veh = self.glimpse_key_fixed_veh.view(batch_size, graph_size, self.opts.n_heads, -1) \
            .permute(2, 0, 1, 3)
        self.glimpse_val_fixed_veh = self.glimpse_val_fixed_veh.view(batch_size, graph_size, self.opts.n_heads, -1) \
            .permute(2, 0, 1, 3)

    def forward(self, embeddings, state, mask, veh):
        batch_size, graph_size, embed_dim = embeddings.size()
        # 得到context    [batch, embeddings_dim + 1]
        context_temp = self.get_context(embeddings, state, veh)
        # 得到query      [batch, 1, embeddings_dim]
        # query到底是拼接还是相加???????????????????????????????????????????????????????????
        query = self.fixed_context + self.project_step_context(context_temp)
        # 加了个全连接，性能变差了5.0669-5.1122
        # query = self.project_query_context(query)

        # 待实现的sdevrp接口
        glimpse_key = self.glimpse_key_fixed
        glimpse_val = self.glimpse_val_fixed
        logit_key = self.logit_key_fixed

        logits_node = self.get_select_p(query, glimpse_key, glimpse_val, logit_key, self.kv_size, mask)

        return logits_node, query

    def select_veh_new(self, state, embeddings, obj, outputs_sequences, path_len, mask_veh_node):
        # Route特征和车辆特征到底选哪些，参考论文再好好想想！！！！！！！！！！！！！！！！！！！！！！！！
        # 加上时间窗后，这些函数都要考虑怎么加上相关信息！！！！！！！！！！！！！！！！
        # 是否需要考虑多车容量和速度的关系？？？？？？？？？？？？？？？？？
        # 选车时也用上mask信息！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 选车时也用上mask信息！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 选车时也用上mask信息！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 选车时也用上mask信息！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        tour_dis = state.lengths  # [batch_size, num_veh]

        if obj == 'min-max':
            SPEED = [1 for _ in range(self.opts.VEHICLE_num)]
        if obj == 'min-sum':
            # 这里是否要改，看论文！！！！！！！！！！！！！！！！！！！！！！！！！
            SPEED = [1 for _ in range(self.opts.VEHICLE_num)]

        batch_size, graph_size, embed_dim = embeddings.size()

        current_loc = state.cur_coord.transpose(0, 1)  # [num_veh, batch_size, 2]
        current_time = state.time
        # 这里用的剩余容量电量
        current_used_capacity = self.opts.hc_VEHICLE_CAPACITY[0] - state.used_capacity
        current_used_Q = self.opts.VEHICLE_Q - state.used_Q
        veh_context = torch.cat(  # [batch_size, 6]
            (
                tour_dis[:, 0].unsqueeze(-1),  # [batch_size, 1]
                current_loc[0, :],  # [batch_size, 2])
                current_time[:, 0].unsqueeze(-1),
                current_used_capacity[:, 0].unsqueeze(-1),
                current_used_Q[:, 0].unsqueeze(-1),
            ),
            -1
        ).unsqueeze(1)  # [batch_size, 6]-># [batch_size, 1, 6]
        for i in range(1, self.opts.VEHICLE_num):
            veh_context_part = torch.cat(  # [batch_size, 6*VEHICLE_num]
                (
                    tour_dis[:, i].unsqueeze(-1),  # [batch_size, 1]
                    current_loc[i, :],  # [batch_size, 2])
                    current_time[:, i].unsqueeze(-1),
                    current_used_capacity[:, i].unsqueeze(-1),
                    current_used_Q[:, i].unsqueeze(-1),
                ),
                -1
            ).unsqueeze(1)
            veh_context = torch.cat((veh_context, veh_context_part), dim=1)

        if path_len == 0:
            mean_tour = torch.zeros([batch_size, self.opts.VEHICLE_num, embed_dim]).float().to(embeddings.device)
        else:
            tour_list_tmp, tour_con_list_tmp = [], []
            for i in range(self.opts.VEHICLE_num):
                tour_list_tmp.append(outputs_sequences[:, i, :path_len])
            for i in range(self.opts.VEHICLE_num):
                ys = torch.gather(
                    embeddings,  # [batch_size, graph_size, embed_dim]
                    1,
                    (tour_list_tmp[i].clone())[..., None].contiguous().expand(batch_size, tour_list_tmp[i].size(-1),
                                                                              embed_dim)
                )
                tour_con_list_tmp.append(ys)

            # 可以试试mean(试了一次效果不行，生成不了可行解)
            mean_tour = torch.max(tour_con_list_tmp[0], dim=1)[0].unsqueeze(1)
            # mean_tour = torch.mean(tour_con_list_tmp[0], dim=1).unsqueeze(1)
            for i in range(1, self.opts.VEHICLE_num):
                mean_tour_part = torch.max(tour_con_list_tmp[i], dim=1)[0].unsqueeze(1)
                # mean_tour_part = torch.mean(tour_con_list_tmp[i], dim=1).unsqueeze(1)
                mean_tour = torch.cat((mean_tour, mean_tour_part), dim=1)

        veh_context = self.FF_veh(veh_context)  # [batch_size, embed_dim]
        tour_context = self.FF_tour(mean_tour)
        # 先试试直接add
        # context = torch.cat((veh_context, tour_context), -1).view(batch_size, embed_dim * 2)
        # 再尝试加一下avg
        tmp_avg = torch.mean(embeddings, dim=1).unsqueeze(1).expand_as(veh_context)
        # context = veh_context + tour_context + tmp_avg
        context = veh_context + tour_context

        # 选车attention-----------------------------------------------------------------------------
        # tmp_k = torch.mean(embeddings, dim=1).unsqueeze(1)
        # tmp_k_veh = tmp_k.view(batch_size, 1, self.opts.n_heads, self.kv_size).permute(2, 0, 1, 3)
        # glimpse_Q_veh = context.view(batch_size, self.opts.VEHICLE_num, self.opts.n_heads, self.kv_size).permute(2, 0, 1, 3)
        # compatibility_veh = torch.matmul(glimpse_Q_veh, tmp_k_veh.transpose(-2, -1)) / math.sqrt(self.kv_size)
        # heads_veh = torch.matmul(torch.softmax(compatibility_veh, dim=-1), tmp_k_veh)
        # glimpse_veh = self.project_out_veh(heads_veh.permute(1, 2, 0, 3).contiguous().view(batch_size, self.opts.VEHICLE_num, embed_dim))
        # logits_veh = torch.matmul(glimpse_veh, tmp_k.transpose(-2, -1)).squeeze(-2) / math.sqrt(self.kv_size)
        # logits_veh = torch.tanh(logits_veh) * self.opts.tanh_clipping
        # logits_veh = logits_veh.squeeze()
        # 选车attention-----------------------------------------------------------------------------
        # 这里没用mask
        glimpse_Q_veh = context.view(batch_size, self.opts.VEHICLE_num, self.opts.n_heads, self.kv_size).permute(2, 0, 1, 3)
        compatibility_veh = torch.matmul(glimpse_Q_veh, self.glimpse_key_fixed_veh.transpose(-2, -1)) / math.sqrt(self.kv_size)
        # 尝试下mask,好像效果不太行
        # compatibility_veh [n_heads, batch_size, VEHICLE_num, graph_size]
        # mask_veh_node [batch_size, VEHICLE_num, graph_size]
        # compatibility_veh[mask_veh_node[None, :, :, :].expand_as(compatibility_veh)] = -math.inf
        heads_veh = torch.matmul(torch.softmax(compatibility_veh, dim=-1), self.glimpse_val_fixed_veh)
        glimpse_veh = self.project_out_veh(
            heads_veh.permute(1, 2, 0, 3).contiguous().view(batch_size, self.opts.VEHICLE_num, embed_dim))
        logits_veh = torch.matmul(glimpse_veh, self.logit_key_fixed_veh.transpose(-2, -1)).squeeze(-2) / math.sqrt(self.kv_size)
        # todo 将tanh替换为clamp，以缓解梯度消失的问题
        # logits_veh = torch.tanh(logits_veh) * self.opts.tanh_clipping
        logits_veh = torch.clamp(logits_veh, -self.opts.tanh_clipping, self.opts.tanh_clipping)
        # 第2个mask还没加，先只试试第一个mask!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # logits_veh[mask_veh_node[None, :, :, :].expand_as(compatibility_veh)] = -math.inf
        logits_veh = self.project_logits_veh(logits_veh).squeeze()
        # print(logits_veh)
        # logits_veh[mask] = -math.inf
        # 选车attention-----------------------------------------------------------------------------
        # 尝试加入图信息
        # avg = torch.mean(embeddings, dim=1).squeeze()
        # context = torch.cat((context, avg), -1).view(batch_size, embed_dim * 3)

        # logits_veh = self.select_veh_embed(context)
        # 这里也加一个梯度裁剪！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 缩放到相同尺度，便于HR！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        ## logits_veh = torch.tanh(logits_veh) * self.opts.tanh_clipping
        return logits_veh, context

    def get_select_p(self, q, k1, v1, k2, kv_size, mask):
        glimpse_Q = q.view(-1, 1, self.opts.n_heads, kv_size).permute(2, 0, 1, 3)
        compatibility = torch.matmul(glimpse_Q, k1.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        compatibility[mask[None, :, None, :].expand_as(compatibility)] = -math.inf
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), v1)
        glimpse = self.project_out(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, 1, self.opts.n_heads * kv_size))
        logits = torch.matmul(glimpse, k2.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))
        # todo 将tanh替换为clamp，以缓解梯度消失的问题
        # logits = torch.tanh(logits) * self.opts.tanh_clipping
        logits = torch.clamp(logits, -self.opts.tanh_clipping, self.opts.tanh_clipping)
        return logits

    def get_context(self, embeddings, state, veh):
        # 还有这里到底是图嵌入+前一节点+容/电量cat还是相加？？？？？？？？？？？？？？？？？？？？？？？？？
        # 到底如何改进
        current_node = state.prev_a
        batch_size, _ = current_node.size()
        # 仅用于去函数波浪线
        yyysss = self.decode_type
        return torch.cat(
            (
                torch.gather(
                    embeddings,
                    1,
                    (current_node[torch.arange(batch_size), veh]).contiguous()
                    .view(batch_size, 1, 1).expand(batch_size, 1, embeddings.size(-1))
                ).view(batch_size, embeddings.size(-1)),  # [batch_size, embed_dim]
                (state.VEHICLE_CAPACITY[None, veh].to(embeddings.device) - state.used_capacity[
                    torch.arange(batch_size), veh]).transpose(0, 1),  # [batch_size, 1]
                (state.VEHICLE_Q - state.used_Q[torch.arange(batch_size), veh])[:, None],  # [batch_size, 1]
                (state.time[torch.arange(batch_size), veh])[:, None]
            ),
            -1
        )

class Trans_Model(nn.Module):
    def __init__(self):
        super(Trans_Model, self).__init__()
        pass

    def forward(self, batch_data):
        return

class Trans_Model_low(Trans_Model):
    def __init__(self,
                 encoder:Trans_Model_Encoder,
                 decoder:Trans_Model_Decoder_low,
                 opts:VRPOption):
        super().__init__()
        self.opts = opts
        self.encoder = encoder
        self.decoder = decoder
        HR_alpha1 = 0.1 * torch.ones(1)
        HR_alpha2 = 0.1 * torch.ones(1)
        self.HR_alpha1 = nn.Parameter(HR_alpha1)
        self.HR_alpha2 = nn.Parameter(HR_alpha2)
        TW_p1 = 0.1 * torch.ones(1)
        self.TW_p1 = nn.Parameter(TW_p1)
        self.max_path_len = self.opts.customer_size * 6

    def forward(self,
                batch_data: torch.Tensor):
        # batch_data = move_to(batch_data, self.device)
        # 编码层---------------------------------------------------------------------------------------------------------
        embeddings, avg = self.encoder(batch_data['loc'], batch_data['demand'], batch_data['start_time'],
                                       batch_data['end_time'], batch_data['service_time'], batch_data['mid_time'])

        # 解码层---------------------------------------------------------------------------------------------------------
        batch_size, graph_size, embed_dim = embeddings.size()
        self.decoder.get_fixed(embeddings, avg)

        # 路径和概率存储列表
        outputs_p = torch.FloatTensor(batch_size, self.max_path_len).to(embeddings.device)
        outputs_sequences = torch.LongTensor(batch_size, self.opts.VEHICLE_num, self.max_path_len).to(embeddings.device)
        outputs_p_veh = torch.FloatTensor(batch_size, self.max_path_len).to(embeddings.device)
        outputs_sequences_veh = torch.LongTensor(batch_size, self.max_path_len).to(embeddings.device)
        state = StateHEVRPSTW(batch_data, embeddings.device, self.opts)
        for path_len in range(self.max_path_len):
            finished_batch, unfinished_batch = state.get_finished()
            if len(unfinished_batch) == 0:
                # 路径点的序列和概率更新
                outputs_p[:, path_len:] = 1
                outputs_sequences[:, :, path_len:] = 0
                # 车辆点的序列和概率更新
                outputs_p_veh[:, path_len:] = 1
                outputs_sequences_veh[:, path_len:] = 0
                # selected目前形状就是[batch, 3]
                # selected = torch.zeros(batch_size, self.opts.VEHICLE_num)
                # 试试改这里会不会解决最后一点的值未计算！！！！！！！！！！！！！！！！！！
                for ys in range(self.opts.VEHICLE_num):
                    tmp_veh = torch.ones(batch_size, dtype=torch.int64, device=embeddings.device)
                    tmp_veh *= ys
                    tmp_selected = state.prev_a.clone()
                    tmp_selected[torch.arange(batch_size), tmp_veh] = 0
                    state.update(tmp_selected, tmp_veh)

                # state.update(selected, veh)
                # tmp_veh = torch.ones(batch_size, dtype=torch.int64, device=embeddings.device)
                # for ys in range(self.opts.VEHICLE_num):
                #     state.update(selected, ys * tmp_veh)

                break
            # 是否重新设计惩罚？？？？？？？？？？？,不同问题如何添加惩罚，目前hc问题的惩罚简单处理了
            if path_len == self.max_path_len - 1:
                _upc = getattr(self.opts, 'unfinished_penalty_coeff', 1000.0)
                state.unfinished_penalty[unfinished_batch, 0] += (
                    _upc * state.num_unvisited_customer[unfinished_batch].float()
                    + _upc * state.num_veh_not_return[unfinished_batch].float()
                )

            # 选车mask
            mask_veh, mask_veh_node = state.get_mask_veh(self.TW_p1)

            # 选车选车---------------------------------------------------------------------------------------------------
            # veh [batch_size]  p_veh [batch_size, 3]
            logits_veh, _ = self.decoder.select_veh_new(state, embeddings, self.opts.hc_obj, outputs_sequences, path_len, mask_veh_node)

            # 优先处理无法继续用的车(这样的话这个车就不会占用概率)
            # 无选车mask---
            if logits_veh.shape != mask_veh.shape:
                logits_veh = logits_veh[None, :]

            logits_veh[mask_veh] = float(-1e20)
            probs_veh = torch.softmax(logits_veh, dim=-1)
            if self.decoder.decode_type == "greedy":
                _, veh = probs_veh.max(1)
            elif self.decoder.decode_type == "sampling":
                veh = probs_veh.multinomial(1).squeeze(-1)
            elif self.decoder.decode_type == "topK":
                # 选车 TopK：候选数取 min(TopK, VEHICLE_num) 防止越界
                topK_size_veh = min(self.opts.TopK, self.opts.VEHICLE_num)
                topK_values_veh, topK_indices_veh = torch.topk(probs_veh, topK_size_veh, dim=-1)
                selected_indices_veh = torch.multinomial(topK_values_veh, 1).squeeze(1)
                veh = torch.gather(topK_indices_veh, 1, selected_indices_veh.unsqueeze(1)).squeeze(1)
            else:
                assert False, "Unknown decode type"
            # 选车选车---------------------------------------------------------------------------------------------------

            # 得到选点mask       [batch, 1, graph_size]
            mask_node = state.get_mask_node(veh, self.TW_p1)
            # 选点选点
            logits_node, _ = self.decoder(embeddings, state, mask_node, veh)

            logits_node[mask_node] = float(-1e20)
            probs_node = torch.softmax(logits_node, dim=-1)
            selected = state.prev_a.clone()
            if self.decoder.decode_type == "greedy":
                # 两个返回值形状均为[batch], 第一个为值，第二个为id
                _, selected[torch.arange(batch_size), veh] = probs_node.max(1)
            elif self.decoder.decode_type == "sampling":
                # 只返回形状为[batch]的id
                selected[torch.arange(batch_size), veh] = probs_node.multinomial(1).squeeze(1)
            elif self.decoder.decode_type == "topK":
                topK_values, topK_indices = torch.topk(probs_node, self.opts.TopK, dim=-1)
                selected_indices = torch.multinomial(topK_values, 1).squeeze(1)
                # 与 greedy/sampling 分支保持一致：只更新当前 veh 对应的 slot，
                # 维持 selected 形状为 [batch_size, VEHICLE_num]，否则 state.update 内
                # 会因 prev_a 退化为 1D 而报 IndexError。
                selected[torch.arange(batch_size), veh] = torch.gather(
                    topK_indices, 1, selected_indices.unsqueeze(1)
                ).squeeze(1)
            else:
                assert False, "Unknown decode type"
            # 更新问题状态
            state.update(selected, veh)
            # 路径点的序列和概率更新
            probs_node = torch.gather(probs_node, 1, selected[torch.arange(batch_size), veh].unsqueeze(1))
            outputs_p[:, path_len] = probs_node[:, 0]
            outputs_p[finished_batch, path_len] = 1
            outputs_sequences[:, :, path_len] = selected
            outputs_sequences[finished_batch, :, path_len] = 0
            # 车辆点的序列和概率更新
            probs_veh = torch.gather(probs_veh, 1, veh.unsqueeze(1))
            outputs_p_veh[:, path_len] = probs_veh[:, 0]
            outputs_p_veh[finished_batch, path_len] = 1
            outputs_sequences_veh[:, path_len] = veh
            outputs_sequences_veh[finished_batch, path_len] = 0

        # 计算cost
        # 底层设置成时间直接不收敛,mb
        # 先试试路程
        # 如果还是无法改成时间的话，把未完成惩罚加到时间里试试？？？？？？？？？？？？？？？。
        _lengths = torch.sum(state.lengths, dim=1)
        _unfinished_penalty = torch.sum(state.unfinished_penalty, dim=1)
        cost = _lengths + _unfinished_penalty
        cost_info = {
            'lengths': _lengths.detach(),
            'time_penalty': torch.zeros_like(_lengths),
            'unfinished_penalty': _unfinished_penalty.detach(),
        }
        # 计算对数似然概率
        # print(self.TW_p1)
        log_outputs_p = torch.log(outputs_p)
        ll = torch.sum(log_outputs_p, dim=1)
        log_outputs_p_veh = torch.log(outputs_p_veh)
        ll_veh = torch.sum(log_outputs_p_veh, dim=1)
        return cost, ll, ll_veh, outputs_sequences, cost_info


class Trans_Model_high(Trans_Model):
    def __init__(self,
                 encoder: Trans_Model_Encoder,
                 decoder: Trans_Model_Decoder_high,
                 opts: VRPOption,
                 model_low: Trans_Model_low):
        super().__init__()
        self.opts = opts
        self.encoder = encoder
        self.decoder = decoder
        self.model_low = model_low
        self.model_low.eval()
        self.model_low.requires_grad_(False)
        TW_p1 = 0.1 * torch.ones(1)
        self.TW_p1 = nn.Parameter(TW_p1)
        self.max_path_len = self.opts.customer_size * 6
        # MoE Gate融合网络
        _fhd = self.opts.fusion_hidden_dim
        _embed = self.opts.embedding_dim

        # MoE Gate: 选点gate，输入为decoder query [batch, embed_dim]
        self.gate_node = nn.Sequential(
            nn.Linear(_embed, _fhd),
            nn.ReLU(),
            nn.Linear(_fhd, 2),  # 2个expert权重
        )

        # MoE Gate: 选车gate，输入为全局context [batch, embed_dim]（mean pooling over vehicles）
        self.gate_veh = nn.Sequential(
            nn.Linear(_embed, _fhd),
            nn.ReLU(),
            nn.Linear(_fhd, 2),
        )

        # 初始化偏向high-level: softmax([b,-b])，b由超参数moe_gate_init_bias控制
        _bias_val = self.opts.moe_gate_init_bias
        nn.init.zeros_(self.gate_node[-1].weight)
        self.gate_node[-1].bias.data = torch.tensor([_bias_val, -_bias_val])
        nn.init.zeros_(self.gate_veh[-1].weight)
        self.gate_veh[-1].bias.data = torch.tensor([_bias_val, -_bias_val])

        self.batch_max_w_low_veh = 0.0
        self.batch_max_w_low_node = 0.0

    def forward(self,
                batch_data:torch.Tensor):
        # batch_data = move_to(batch_data, self.device)
        # 编码层---------------------------------------------------------------------------------------------------------
        embeddings, avg = self.encoder(batch_data['loc'], batch_data['demand'], batch_data['start_time'],
                                       batch_data['end_time'], batch_data['service_time'], batch_data['mid_time'])
        # HR的话编码器需要用同一个吗???????????????????????????????????????
        with torch.no_grad():
            embeddings_low, avg_low = self.model_low.encoder(batch_data['loc'], batch_data['demand'],
                                                        batch_data['start_time'], batch_data['end_time'],
                                                        batch_data['service_time'], batch_data['mid_time'])
        # 解码层---------------------------------------------------------------------------------------------------------
        batch_size, graph_size, embed_dim = embeddings.size()
        self.decoder.get_fixed(embeddings, avg)
        with torch.no_grad():
            self.model_low.decoder.get_fixed(embeddings_low, avg_low)
        # 路径和概率存储列表
        outputs_p = torch.FloatTensor(batch_size, self.max_path_len).to(embeddings.device)
        outputs_sequences = torch.LongTensor(batch_size, self.opts.VEHICLE_num, self.max_path_len).to(embeddings.device)
        outputs_p_veh = torch.FloatTensor(batch_size, self.max_path_len).to(embeddings.device)
        outputs_sequences_veh = torch.LongTensor(batch_size, self.max_path_len).to(embeddings.device)
        # --- Reward-to-Go: 记录每步增量 cost ---
        step_costs = torch.zeros(batch_size, self.max_path_len, device=embeddings.device)
        prev_lengths_sum = torch.zeros(batch_size, device=embeddings.device)
        prev_time_penalty_sum = torch.zeros(batch_size, device=embeddings.device)
        state = StateHEVRPSTW(batch_data, embeddings.device, self.opts)
        max_w_low_veh = 0.0
        max_w_low_node = 0.0
        for path_len in range(self.max_path_len):
            finished_batch, unfinished_batch = state.get_finished()
            if len(unfinished_batch) == 0:
                # 路径点的序列和概率更新
                outputs_p[:, path_len:] = 1
                outputs_sequences[:, :, path_len:] = 0
                # 车辆点的序列和概率更新
                outputs_p_veh[:, path_len:] = 1
                outputs_sequences_veh[:, path_len:] = 0
                # selected目前形状就是[batch, 3]
                # selected = torch.zeros(batch_size, self.opts.VEHICLE_num)
                # 试试改这里会不会解决最后一点的值未计算！！！！！！！！！！！！！！！！！！
                for ys in range(self.opts.VEHICLE_num):
                    tmp_veh = torch.ones(batch_size, dtype=torch.int64, device=embeddings.device)
                    tmp_veh *= ys
                    tmp_selected = state.prev_a.clone()
                    tmp_selected[torch.arange(batch_size), tmp_veh] = 0
                    state.update(tmp_selected, tmp_veh)

                # state.update(selected, veh)
                # tmp_veh = torch.ones(batch_size, dtype=torch.int64, device=embeddings.device)
                # for ys in range(self.opts.VEHICLE_num):
                #     state.update(selected, ys * tmp_veh)

                break
            # 是否重新设计惩罚？？？？？？？？？？？,不同问题如何添加惩罚，目前hc问题的惩罚简单处理了
            if path_len == self.max_path_len - 1:
                _upc = getattr(self.opts, 'unfinished_penalty_coeff', 1000.0)
                state.unfinished_penalty[unfinished_batch, 0] += (
                    _upc * state.num_unvisited_customer[unfinished_batch].float()
                    + _upc * state.num_veh_not_return[unfinished_batch].float()
                )

            # 选车mask
            mask_veh, mask_veh_node = state.get_mask_veh(self.TW_p1)

            # 选车选车---------------------------------------------------------------------------------------------------
            # veh [batch_size]  p_veh [batch_size, 3]
            logits_veh, ctx_veh = self.decoder.select_veh_new(state, embeddings, self.opts.hc_obj, outputs_sequences, path_len, mask_veh_node)
            with torch.no_grad():
                logits_veh_low, _ = self.model_low.decoder.select_veh_new(state, embeddings_low, self.opts.hc_obj,
                                                                  outputs_sequences, path_len, mask_veh_node)
            # MoE gate: per-step全局权重（对车辆维度mean pooling）
            ctx_veh_global = ctx_veh.mean(dim=1)  # [batch, embed_dim] — 对车辆维度做mean pooling
            gate_weights_veh = F.softmax(self.gate_veh(ctx_veh_global), dim=-1)  # [batch, 2]
            logits_veh = gate_weights_veh[:, 0:1] * logits_veh + gate_weights_veh[:, 1:2] * logits_veh_low
            max_w_low_veh = max(max_w_low_veh, gate_weights_veh[:, 1].max().item())

            # 优先处理无法继续用的车(这样的话这个车就不会占用概率)
            # 无选车mask---
            if logits_veh.shape != mask_veh.shape:
                logits_veh = logits_veh[None, :]

            logits_veh[mask_veh] = float(-1e20)
            probs_veh = torch.softmax(logits_veh, dim=-1)
            if self.decoder.decode_type == "greedy":
                _, veh = probs_veh.max(1)
            elif self.decoder.decode_type == "sampling":
                veh = probs_veh.multinomial(1).squeeze(-1)
            elif self.decoder.decode_type == "topK":
                # 选车 TopK：候选数取 min(TopK, VEHICLE_num) 防止越界
                topK_size_veh = min(self.opts.TopK, self.opts.VEHICLE_num)
                topK_values_veh, topK_indices_veh = torch.topk(probs_veh, topK_size_veh, dim=-1)
                selected_indices_veh = torch.multinomial(topK_values_veh, 1).squeeze(1)
                veh = torch.gather(topK_indices_veh, 1, selected_indices_veh.unsqueeze(1)).squeeze(1)
            else:
                assert False, "Unknown decode type"
            # 选车选车---------------------------------------------------------------------------------------------------

            # 得到选点mask       [batch, 1, graph_size]
            mask_node = state.get_mask_node(veh, self.TW_p1)
            # 选点选点
            logits_node, state_node = self.decoder(embeddings, state, mask_node, veh)
            with torch.no_grad():
                logits_node_low, _ = self.model_low.decoder(embeddings_low, state, mask_node, veh)
            # MoE gate: per-step权重
            gate_weights_node = F.softmax(self.gate_node(state_node), dim=-1)  # [batch, 2]
            logits_node = gate_weights_node[:, 0:1] * logits_node + gate_weights_node[:, 1:2] * logits_node_low
            max_w_low_node = max(max_w_low_node, gate_weights_node[:, 1].max().item())

            logits_node[mask_node] = float(-1e20)
            probs_node = torch.softmax(logits_node, dim=-1)
            selected = state.prev_a.clone()
            if self.decoder.decode_type == "greedy":
                # 两个返回值形状均为[batch], 第一个为值，第二个为id
                _, selected[torch.arange(batch_size), veh] = probs_node.max(1)
            elif self.decoder.decode_type == "sampling":
                # 只返回形状为[batch]的id
                selected[torch.arange(batch_size), veh] = probs_node.multinomial(1).squeeze(1)
            elif self.decoder.decode_type == "topK":
                topK_values, topK_indices = torch.topk(probs_node, self.opts.TopK, dim=-1)
                selected_indices = torch.multinomial(topK_values, 1).squeeze(1)
                # 与 greedy/sampling 分支保持一致：只更新当前 veh 对应的 slot，
                # 维持 selected 形状为 [batch_size, VEHICLE_num]，否则 state.update 内
                # 会因 prev_a 退化为 1D 而报 IndexError。
                selected[torch.arange(batch_size), veh] = torch.gather(
                    topK_indices, 1, selected_indices.unsqueeze(1)
                ).squeeze(1)
            else:
                assert False, "Unknown decode type"
            # 更新问题状态
            state.update(selected, veh)
            # --- Reward-to-Go: 记录本步增量 cost（路径 + 时间惩罚）---
            cur_lengths_sum = torch.sum(state.lengths, dim=1)
            cur_time_penalty_sum = torch.sum(state.time_penalty, dim=1)
            step_costs[:, path_len] = (
                (cur_lengths_sum - prev_lengths_sum)
                + (cur_time_penalty_sum - prev_time_penalty_sum)
            )
            prev_lengths_sum = cur_lengths_sum
            prev_time_penalty_sum = cur_time_penalty_sum
            # 路径点的序列和概率更新
            probs_node = torch.gather(probs_node, 1, selected[torch.arange(batch_size), veh].unsqueeze(1))
            outputs_p[:, path_len] = probs_node[:, 0]
            outputs_p[finished_batch, path_len] = 1
            outputs_sequences[:, :, path_len] = selected
            outputs_sequences[finished_batch, :, path_len] = 0
            # 车辆点的序列和概率更新
            probs_veh = torch.gather(probs_veh, 1, veh.unsqueeze(1))
            outputs_p_veh[:, path_len] = probs_veh[:, 0]
            outputs_p_veh[finished_batch, path_len] = 1
            outputs_sequences_veh[:, path_len] = veh
            outputs_sequences_veh[finished_batch, path_len] = 0

        self.batch_max_w_low_veh = max_w_low_veh
        self.batch_max_w_low_node = max_w_low_node

        # 计算cost
        _lengths = torch.sum(state.lengths, dim=1)
        _time_penalty = torch.sum(state.time_penalty, dim=1)
        _unfinished_penalty = torch.sum(state.unfinished_penalty, dim=1)
        cost = _lengths + _time_penalty + _unfinished_penalty

        # --- Reward-to-Go: 将 unfinished_penalty 追加到最后一个有效步 ---
        # 找到每个样本最后一个 step_cost > 0 的位置
        active_mask = (step_costs != 0).float()  # [batch, T]
        # 若全为0（极端情况），默认加到第0步
        last_active_idx = torch.clamp(
            active_mask.sum(dim=1).long() - 1, min=0
        )  # [batch]
        step_costs[torch.arange(batch_size, device=embeddings.device), last_active_idx] += (
            _unfinished_penalty.detach()
        )

        cost_info = {
            'lengths': _lengths.detach(),
            'time_penalty': _time_penalty.detach(),
            'unfinished_penalty': _unfinished_penalty.detach(),
            'step_costs': step_costs.detach(),  # [batch, T] 每步增量cost（含unfinished）
        }
        # 计算对数似然概率
        log_outputs_p = torch.log(outputs_p)
        ll = torch.sum(log_outputs_p, dim=1)
        log_outputs_p_veh = torch.log(outputs_p_veh)
        ll_veh = torch.sum(log_outputs_p_veh, dim=1)
        # per-step log_prob 存入 cost_info（保留梯度，用于 Reward-to-Go loss）
        cost_info['log_prob_per_step'] = log_outputs_p + log_outputs_p_veh  # [batch, T]
        return cost, ll, ll_veh, outputs_sequences, cost_info
