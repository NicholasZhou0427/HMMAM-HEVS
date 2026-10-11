import copy
from pathlib import Path

import torch
from torch.utils.data import Dataset
from functions import load_pt


class HEVRPSTWDatasetFromFile(Dataset):
    """读取已发布的数据文件；不提供在线生成。"""

    def __init__(self, file_paths, max_samples=None, customer_size=None):
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        if not file_paths:
            raise ValueError('必须指定至少一个 .pt 数据文件')
        if max_samples is not None and max_samples < 1:
            raise ValueError('max_samples 必须大于零')
        self.data = []
        required = {'loc', 'demand', 'start_time', 'end_time', 'service_time', 'mid_time'}
        for path in file_paths:
            samples = load_pt(path, map_location='cpu')
            if not isinstance(samples, list):
                raise ValueError(f'{path}: 数据应为样本字典列表')
            if max_samples is not None:
                samples = samples[:max_samples - len(self.data)]
            for sample in samples:
                if not isinstance(sample, dict) or not required.issubset(sample):
                    raise ValueError(f'{path}: 样本缺少必要字段')
                nodes = sample['loc'].shape[0]
                if sample['loc'].shape != (nodes, 2) or nodes <= 21:
                    raise ValueError(f'{path}: loc 形状应为 [21 + 客户数, 2]')
                if customer_size is not None and nodes != customer_size + 21:
                    raise ValueError(f'{path}: 数据客户数与配置不符')
                for key in required - {'loc'}:
                    if sample[key].shape != (nodes, 1):
                        raise ValueError(f'{path}: {key} 形状应为 [{nodes}, 1]')
                if not all(torch.isfinite(sample[key]).all() for key in required):
                    raise ValueError(f'{path}: 样本包含非有限值')
                self.data.append(sample)
            if max_samples is not None and len(self.data) >= max_samples:
                break
        if not self.data:
            raise ValueError('数据文件没有样本')
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


class StateHEVRPSTW:

    def __init__(self, input, device, opts):
        super(StateHEVRPSTW, self).__init__()

        self.coords = input['loc']
        self.demand = input['demand']
        self.start_time = input['start_time']
        self.end_time = input['end_time']
        self.service_time = input['service_time']

        self.device = device
        self.opts = opts
        self.batch_size = self.coords.size(0)
        self.graph_size = self.coords.size(1)
        self.finished_batch = []
        self.arange_batch = torch.arange(self.batch_size, device=self.device)

        # 车辆超参数
        self.VEHICLE_CAPACITY = torch.tensor(self.opts.hc_VEHICLE_CAPACITY, device=self.device)
        self.VEHICLE_num = len(self.VEHICLE_CAPACITY)
        self.VEHICLE_Q = torch.tensor(self.opts.VEHICLE_Q, device=self.device)
        self.Refuel_rate = torch.tensor(self.opts.Refuel_rate, device=self.device)

        self.prev_a = torch.zeros(self.batch_size, self.VEHICLE_num, dtype=torch.long, device=self.device)
        self.used_capacity = torch.zeros(self.batch_size, self.VEHICLE_num, device=self.device)
        # 电量
        self.used_Q = torch.zeros(self.batch_size, self.VEHICLE_num, device=self.device)

        self.is_visited = torch.zeros(self.batch_size, self.graph_size, dtype=torch.bool, device=self.device)
        self.lengths = torch.zeros(self.batch_size, self.VEHICLE_num, device=self.device)
        self.cur_coord = self.coords[:, 0, :][:, None, :].expand(self.batch_size, self.VEHICLE_num, -1)

        self.dist = (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)
        self.dist_max = torch.max(self.dist)

        # 时间
        self.time = torch.zeros(self.batch_size, self.VEHICLE_num, device=self.device)
        self.time_penalty = torch.zeros(self.batch_size, self.VEHICLE_num, device=self.device)
        self.max_TW = self.end_time.squeeze(-1)[0][0]
        # 未完成惩罚
        self.unfinished_penalty = torch.zeros(self.batch_size, 1, device=self.device)
        # 充电桩惩罚（基于剩余电量，惩罚不必要的充电行为）
        self.charging_penalty = torch.zeros(self.batch_size, self.VEHICLE_num, device=self.device)

    # 编码原则：所有涉及到多车的都用veh获取，从而形式上跟单车统一！！！！！！！！！！！！！！！！！！！！
    def update(self, selected, veh):
        self.prev_a = selected
        self.prev_a = self.prev_a.long().to(self.device)
        self.is_visited = self.is_visited.scatter(-1, self.prev_a[self.arange_batch, veh][:, None], 1)
        cur_coord = torch.gather(self.coords, 1,
                                 self.prev_a.view(self.batch_size, self.VEHICLE_num, 1).expand(self.batch_size,
                                                                                               self.VEHICLE_num,
                                                                                               self.coords.size(-1))
                                 )
        # cur_coord = cur_coord.view(self.batch_size, self.coords.size(2))
        tmp_dis1 = ((cur_coord - self.cur_coord).norm(p=2, dim=-1)).view(-1, self.VEHICLE_num)  # (batch, v_num)
        self.cur_coord = cur_coord
        selected_demand_broad = torch.gather(self.demand, 1,
                                             self.prev_a[self.arange_batch, veh].view(self.batch_size, 1,
                                                                                      1).expand(
                                                 self.batch_size,
                                                 self.VEHICLE_num,
                                                 self.demand.size(-1))
                                             ).squeeze(2)
        # 更新容量
        self.used_capacity[self.arange_batch, veh] += selected_demand_broad[self.arange_batch, veh]

        # 电量(先加上充电时间再充电！！！！！！！！！！！！！！！！！！！！！！！！！！！)
        # self.used_Q = self.used_Q + tmp_dis1
        self.used_Q[self.arange_batch, veh] = self.used_Q[self.arange_batch, veh] + tmp_dis1[self.arange_batch, veh]
        # tmp_time = (self.VEHICLE_Q - self.used_Q) / self.Refuel_rate
        # 到底是×还是÷？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        # tmp_time = self.used_Q / self.Refuel_rate
        tmp_time = self.used_Q[self.arange_batch, veh] * self.Refuel_rate
        # for ys in range(self.batch_size):
        #     if not (self.prev_a[ys][veh[ys]] in [i for i in range(1, 21)]):
        #         tmp_time[ys] = 0
        tmp_charge_station_id = torch.tensor([i for i in range(1, 21)]).to(self.device)
        charge_indices = torch.isin(self.prev_a[self.arange_batch, veh], tmp_charge_station_id)
        tmp_time[~charge_indices] = 0
        self.time[self.arange_batch, veh] += tmp_time

        # 充电桩惩罚：到达充电桩时，剩余电量越多惩罚越大（不必要的充电行为）
        # penalty = coeff * (VEHICLE_Q - used_Q) / VEHICLE_Q，仅对进入充电桩的车计算
        remaining_ratio = (self.VEHICLE_Q - self.used_Q[self.arange_batch, veh]) / self.VEHICLE_Q
        cs_penalty = self.opts.charging_penalty_coeff * remaining_ratio
        cs_penalty[~charge_indices] = 0  # 非充电桩节点不惩罚
        self.charging_penalty[self.arange_batch, veh] += cs_penalty

        for i in range(1, 21):
            self.used_Q[self.arange_batch, veh] *= (self.prev_a[self.arange_batch, veh] != i).float()

        # 时间窗（看看对不对，再看看是否过于简单了）
        # 软时间---------------------------------------------------------------------------------------------------------
        # 路程
        self.lengths[self.arange_batch, veh] += tmp_dis1[self.arange_batch, veh]  # (batch_dim, 10)
        # 路程时间
        self.time[self.arange_batch, veh] += tmp_dis1[self.arange_batch, veh]
        # 记录访问该节点的车辆
        # 软时间窗惩罚
        tmp_t1 = self.time[self.arange_batch, veh]
        tmp_t2 = self.start_time[self.arange_batch, self.prev_a[self.arange_batch, veh]].squeeze(-1)
        tmp_t3 = self.end_time[self.arange_batch, self.prev_a[self.arange_batch, veh]].squeeze(-1)
        time_penalty_start = torch.lt(tmp_t1, tmp_t2).float() * (tmp_t2 - tmp_t1)
        time_penalty_start *= self.opts.STW_penalty_start
        time_penalty_end = torch.lt(tmp_t3, tmp_t1).float() * (tmp_t1 - tmp_t3)
        time_penalty_end *= self.opts.STW_penalty_end

        self.time_penalty[self.arange_batch, veh] += (time_penalty_start + time_penalty_end)
        # 等待时间
        self.time[self.arange_batch, veh] += (torch.lt(tmp_t1, tmp_t2).float() * (tmp_t2 - tmp_t1))

        # 服务时间
        tmp_service = self.service_time[self.arange_batch, self.prev_a[self.arange_batch, veh]].squeeze(-1)
        self.time[self.arange_batch, veh] += tmp_service

        # 计算完惩罚后，再计算仓库点充电时间（处理车辆回仓库后又继续出发的情况）！！！！！！！！！！！！（待验证）
        tmp_time = self.used_Q[self.arange_batch, veh] * self.Refuel_rate
        tmp_depot_station_id = self.prev_a[self.arange_batch, veh] == 0
        tmp_time[~tmp_depot_station_id] = 0
        self.time[self.arange_batch, veh] += tmp_time
        # 判断仓库
        # self.time[self.arange_batch, veh] *= (self.prev_a[self.arange_batch, veh] != 0).float()
        self.used_capacity[self.arange_batch, veh] *= (self.prev_a[self.arange_batch, veh] != 0).float()
        self.used_Q[self.arange_batch, veh] *= (self.prev_a[self.arange_batch, veh] != 0).float()

    def get_finished(self):
        is_visited_slice = self.is_visited[:, 21:].all(dim=1)
        tmp_dist = torch.gather(self.dist[:, :, 0], 1, self.prev_a)
        used_Q_slice = (self.used_Q + tmp_dist) <= self.VEHICLE_Q
        is_valid_Q = used_Q_slice.sum(dim=1) == self.VEHICLE_num
        mask = is_visited_slice & is_valid_Q
        tmp_finished_batch1 = torch.nonzero(mask).squeeze(dim=1)
        unfinished_batch2 = torch.nonzero(~mask).squeeze(dim=1)

        ## todo：加入了“未访问客户数”和“未回仓车辆数”，待验证
        self.num_unvisited_customer = (self.is_visited[:, 21:] == 0).float().sum(dim=1)
        self.num_veh_not_return     = (self.VEHICLE_num - used_Q_slice.sum(dim=1))

        self.finished_batch = tmp_finished_batch1

        return self.finished_batch, unfinished_batch2

    def get_mask_veh(self, TW_p1):
        # 这个函数也需要好好设计！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        mask_veh = torch.zeros(self.batch_size, self.VEHICLE_num, dtype=torch.bool, device=self.device)
        mask_veh_node = torch.zeros(self.batch_size, self.VEHICLE_num, self.graph_size, dtype=torch.bool, device=self.device)
        for id_veh in range(self.VEHICLE_num):
            tmp_veh = [id_veh] * self.batch_size
            tmp_mask = self.get_mask_node(tmp_veh, TW_p1)
            # for id_batch in range(self.batch_size):
            #     tmp_mask_part = tmp_mask[id_batch][21:]
            #     if torch.all(tmp_mask_part) and self.prev_a[id_batch][id_veh] in [0]:
            #         mask_veh[id_batch][id_veh] = 1
            tmp_mask_part = tmp_mask[:, 21:]
            valid_conditions = torch.all(tmp_mask_part, dim=-1) & (self.prev_a[self.arange_batch, tmp_veh] == 0)
            mask_veh[:, id_veh] = valid_conditions
            # 把mask_veh_node也更新一下，用于选车时用到
            mask_veh_node[:, id_veh, :] = tmp_mask

        return mask_veh, mask_veh_node

        # mask_veh_P = torch.zeros(self.batch_size, self.VEHICLE_num, dtype=torch.bool, device=self.device)
        # mask_veh_P[:, 1:] = 1
        # mask_veh_N = torch.zeros(self.batch_size, self.VEHICLE_num, dtype=torch.bool, device=self.device)
        # mask_veh_N[:, 0] = 1
        # return mask_veh_P, mask_veh_N

    def get_mask_node(self, veh, TW_p1):
        """
                Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
                remaining capacity. 0 = feasible, 1 = infeasible
                Forbids to visit depot twice in a row, unless all nodes have been visited
                :return:
        """
        # 令仓库和充电桩不受访问次数限制（但是任务完成前不能待在原地）
        self.is_visited[:, 0:21] = torch.zeros(21)
        self.is_visited = self.is_visited.scatter(-1, self.prev_a[self.arange_batch, veh][:, None], 1)
        # mask1: 访问次数限制
        mask1 = self.is_visited > 0
        # mask2: 容量限制
        # 这个mask2长得有点怪！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # mask2 = self.demand[:, :, 0] + self.used_capacity > self.VEHICLE_CAPACITY
        mask2 = self.demand[:, :, 0] + self.used_capacity[self.arange_batch, veh][:, None] > \
                self.VEHICLE_CAPACITY[None, veh].transpose(0, 1)

        # mask3: 电量限制
        # tmp_dis2: 所有批次下当前点到其他点的距离
        tmp_dis2 = self.dist[self.arange_batch, self.prev_a[self.arange_batch, veh], :]

        mask3 = tmp_dis2 + self.used_Q[self.arange_batch, veh][:, None] > self.VEHICLE_Q
        # mask4: 安全限制
        tmp_Q = tmp_dis2 + self.used_Q[self.arange_batch, veh][:, None]
        # 下面这行加不加都行
        tmp_Q[:, :21] = 0
        mask4 = tmp_Q + self.dist[:, :, 0] <= self.VEHICLE_Q
        for i in range(1, 21):
            mask4 += (tmp_Q + self.dist[:, i, :] <= self.VEHICLE_Q)
        mask4 = ~mask4
        # mask5: 时间窗
        # arrive_time = self.time[self.arange_batch, veh][:, None] + tmp_dis2
        # mask5 = arrive_time > self.end_time.squeeze(-1)
        # mask5[:, 21:] = False

        # 时间安全(参考电量安全)！！！！！！！！！！
        # 如果之前就在仓库点，那么由于返回仓库点充电时间，mask_TW_safe可能会将仓库点mask掉，这里处理
        mask_TW_safe = self.get_mask_TW_safe(veh)
        mask_TW_safe_part = self.prev_a[self.arange_batch, veh] == 0
        mask_TW_safe[mask_TW_safe_part, 0] = False

        # mask_cs_suppress: 电量充足时屏蔽充电桩
        # 基于综合约束（mask1+mask2+mask3+mask4+mask_TW_safe）的客户可达性判断
        mask_cs_suppress = torch.zeros(self.batch_size, self.graph_size, dtype=torch.bool, device=self.device)
        # 综合客户点的所有约束（不含 mask_cs_suppress 本身）
        combined_customer_mask = (mask1 + mask2 + mask3 + mask4 + mask_TW_safe)[:, 21:] > 0
        # 真正可达的客户点：未被任何约束屏蔽
        reachable_customers = ~combined_customer_mask  # [batch_size, customer_size]
        num_reachable = reachable_customers.float().sum(dim=1)  # [batch_size]
        num_unvisited = (~self.is_visited[:, 21:]).float().sum(dim=1)
        # 条件：可达客户占未访问客户比例超过阈值 AND 未访问客户足够多
        ratio = num_reachable / (num_unvisited + 1e-8)
        suppress_batch = (ratio > self.opts.cs_suppress_threshold) & \
                         (num_unvisited >= self.opts.cs_suppress_min_unvisited)
        mask_cs_suppress[suppress_batch, 1:21] = True

        mask = mask1 + mask2 + mask3 + mask4 + mask_cs_suppress + mask_TW_safe

        mask[self.finished_batch, 0] = False
        mask[self.finished_batch, 1:] = True

        # 判断仓库点
        # for id_batch in range(self.batch_size):
        #     if self.prev_a[id_batch][veh[id_batch]] in [0]:
        #         if torch.all(mask[id_batch][21:]):
        #             mask[id_batch, 0] = False
        tmp_batch_id = (self.prev_a[self.arange_batch, veh] == 0) & torch.all(mask[:, 21:], dim=-1)
        mask[tmp_batch_id, 0] = False
        mask[tmp_batch_id, 1:] = True
        # 判断充电桩
        # 如果当前在充电桩，且已没有客户点，那就直接回仓库
        tmp_charge_station_id = torch.tensor([i for i in range(1, 21)]).to(self.device)
        charge_indices = torch.isin(self.prev_a[self.arange_batch, veh], tmp_charge_station_id)
        tmp_batch_id = charge_indices & torch.all(mask[:, 21:], dim=-1)
        mask[tmp_batch_id, 0] = False
        mask[tmp_batch_id, 1:] = True

        return mask

    def __getitem__(self, key):
        return

    # 整理思路：到下一点后有足够时间返回仓库点
    def get_mask_TW_safe(self, veh):
        """
        分解：
        1.假设这一步时所有点仍可选（违反约束的点与上述mask相加即可去除） [self.batch_size, self.graph_size]
        判断当前电量能否直接到仓库(第一次移动到充电桩点的/电量充足的都能回)
        2.少量未能直接返回仓库的客户点需要先移动到充电桩节点然后再到仓库
        只要每个点有一条路径能保证时间安全，这个点就是False!
        """
        # 将到达该点后无法及时返回仓库的点mask掉
        mask_TW_safe = torch.zeros(self.batch_size, self.graph_size, dtype=torch.bool, device=self.device)

        cur_time = copy.deepcopy(self.time)  # [batch_size, VEHICLE_num]
        cur_time = cur_time[self.arange_batch, veh][:, None]  # [batch_size, 1]
        cur_time = cur_time.expand(self.batch_size, self.graph_size).clone()  # [batch_size, graph_size]
        cur_Q = copy.deepcopy(self.used_Q)  # [batch_size, VEHICLE_num]
        cur_Q = cur_Q[self.arange_batch, veh][:, None]  # [batch_size, 1]
        cur_Q = cur_Q.expand(self.batch_size, self.graph_size).clone()  # [batch_size, graph_size]
        # 第一次移动(到达下一点)------------------------------------------------------------------------------------------------
        # tmp_dis2: 所有批次下当前点到其他点的距离   [batch_size, graph_size]
        # tmp_dis2 = (self.coords - self.cur_coord.unsqueeze(1)).norm(p=2, dim=-1)
        tmp_dis2 = self.dist[self.arange_batch, self.prev_a[self.arange_batch, veh], :]
        # 时间更新：路程时间 等待时间 充电时间 服务时间
        cur_time += tmp_dis2
        cur_Q += tmp_dis2
        # 这里没有判断是否超后时间窗，对硬时间窗问题是否有影响，如果带上mask5应该就无影响！！！！！！！！！！！！
        cur_time = torch.max(self.start_time.squeeze(-1), cur_time)
        tmp_time = cur_Q * self.Refuel_rate
        tmp_time[:, 0] = 0
        tmp_time[:, 21:] = 0
        cur_time += tmp_time
        cur_time += self.service_time.squeeze(-1)
        # 电量更新
        cur_Q[:, :21] = 0

        # 第二次移动------------------------------------------------------------------------------------------------
        # 判断当前电量能否直接到仓库(第一次移动到充电桩点的/电量充足的都能回)
        # 所有点到仓库点的路程：self.dist[:, 0, :]   [batch_size, graph_size]
        # 情况1：直达时间已超过
        tmp_mask1 = cur_time + self.dist[:, 0, :] > self.max_TW
        mask_TW_safe[tmp_mask1] = True
        # for id_batch in range(self.batch_size):
        #     for id_node in range(self.graph_size):
        #         part2_Q = cur_Q[id_batch][id_node]
        #         part2_time = cur_time[id_batch][id_node]
        #         if not tmp_mask1[id_batch][id_node]:
        #             # 情况2：直达时间未超过，且可以直达
        #             if (part2_Q + self.dist[id_batch, id_node, 0]) <= self.VEHICLE_Q:
        #                 pass
        #             # 情况3：直达时间未超过，不能直达
        #             else:
        #                 # 如果不存在任一一条路径使及时返回，则这个点mask掉
        #                 can_return_D = False
        #                 for id_Charging_Station in range(1, 21):
        #                     new_part2_Q = part2_Q + self.dist[id_batch, id_node, id_Charging_Station]
        #                     if new_part2_Q <= self.VEHICLE_Q:
        #                         new_part2_time = part2_time + self.dist[id_batch, id_node, id_Charging_Station]
        #                         new_part2_time += new_part2_Q * self.Refuel_rate
        #                         if (new_part2_time + self.dist[id_batch, id_Charging_Station, 0]) <= self.max_TW:
        #                             can_return_D = True
        #                             break
        #                 if not can_return_D:
        #                     mask_TW_safe[id_batch][id_node] = True
        can_return_D = torch.zeros(self.batch_size, self.graph_size, dtype=torch.bool, device=self.device)
        can_return_D += (((cur_Q + self.dist[:, :, 0]) <= self.VEHICLE_Q) & (cur_time + self.dist[:, 0, :] <= self.max_TW))
        for id_Charging_Station in range(21):
            new_part2_Q = cur_Q + self.dist[:, :, id_Charging_Station]
            tmp_ys2 = self.dist[:, id_Charging_Station, 0].unsqueeze(-1).expand(self.batch_size, self.graph_size)
            # 仓库点不用充电时间
            if id_Charging_Station == 0:
                can_return_D += (new_part2_Q <= self.VEHICLE_Q) & (
                        (cur_time + self.dist[:, :, id_Charging_Station] + tmp_ys2) <= self.max_TW)
            else:
                can_return_D += (new_part2_Q <= self.VEHICLE_Q) & (
                        (cur_time + self.dist[:, :, id_Charging_Station] + new_part2_Q * self.Refuel_rate +
                         tmp_ys2) <= self.max_TW)
        can_return_D = ~can_return_D
        mask_TW_safe[can_return_D] = True
        return mask_TW_safe
