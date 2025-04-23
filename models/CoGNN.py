import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, Dropout, LayerNorm, Identity
import torch.nn.functional as F
from typing import Tuple
import numpy as np
import torch.nn as nn
from torch_scatter import scatter_mean  # 用于计算全局上下文
from helpers.classes import GumbelArgs, EnvArgs, ActionNetArgs, Pool, DataSetEncoders
from models.temp import TempSoftPlus
from models.action import ActionNet


class CoGNN(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActionNetArgs, pool: Pool):
        super(CoGNN, self).__init__()
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args.env_dim)
        self.temp = gumbel_args.temp

        self.num_layers = env_args.num_layers
        self.env_net = env_args.load_net()
        self.use_encoders = env_args.dataset_encoders.use_encoders()

        layer_norm_cls = LayerNorm if env_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(env_args.env_dim)
        self.skip = env_args.skip
        self.dropout = Dropout(p=env_args.dropout)
        self.drop_ratio = env_args.dropout
        self.act = env_args.act_type.get()

        history_dim = min(4, self.num_layers) * 4  # 只保留最近4层或更少的历史
        self.in_act_net = ActionNet(action_args=action_args,history_dim=history_dim)
        self.out_act_net = ActionNet(action_args=action_args,history_dim=history_dim)

        # Encoder types
        self.dataset_encoder = env_args.dataset_encoders
        self.env_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=env_args.env_dim, model_type=env_args.model_type)
        self.act_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=action_args.hidden_dim, model_type=action_args.model_type)

        # Pooling function to generate whole-graph embeddings
        self.pooling = pool.get()
##########################################修改：全局上下文#########################
        # 添加全局上下文处理组件
        self.context_encoder = nn.Sequential(
            nn.Linear(env_args.env_dim, env_args.env_dim),
            nn.BatchNorm1d(env_args.env_dim),
            nn.ReLU(),
            nn.Linear(env_args.env_dim, env_args.env_dim)
        )

        # 添加动作精细化组件
        self.action_refiner = nn.Sequential(
            nn.Linear(2*env_args.env_dim, env_args.env_dim),
            nn.BatchNorm1d(env_args.env_dim),
            nn.ReLU(),
            nn.Linear(int(env_args.env_dim), 4))

    def forward(self, x: Tensor, edge_index: Adj, pestat, edge_attr: OptTensor = None, batch: OptTensor = None,
                edge_ratio_node_mask: OptTensor = None, action_history: OptTensor = None) -> Tuple[Tensor, Tensor]:
        result = 0
        num_nodes = x.size(0)

        # 初始化或使用传入的决策历史
        if action_history is None:
            # 初始化为全零矩阵
            action_history = torch.zeros(num_nodes, self.num_layers * 4, device=x.device)

        # 创建历史副本用于更新
        updated_history = action_history.clone()

        calc_stats = edge_ratio_node_mask is not None
        if calc_stats:
            edge_ratio_edge_mask = edge_ratio_node_mask[edge_index[0]] & edge_ratio_node_mask[edge_index[1]]
            edge_ratio_list = []

        # bond encode
        if edge_attr is None or self.env_bond_encoder is None:
            env_edge_embedding = None
        else:
            env_edge_embedding = self.env_bond_encoder(edge_attr)
        if edge_attr is None or self.act_bond_encoder is None:
            act_edge_embedding = None
        else:
            act_edge_embedding = self.act_bond_encoder(edge_attr)

        # node encode  
        x = self.env_net[0](x, pestat)  # (N, F) encoder
        if not self.use_encoders:
            x = self.dropout(x)
            x = self.act(x)


        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)

            ###########################################33
            # # 计算全局上下文信息
            # if batch is not None:
            #     graph_context = scatter_mean(x, batch, dim=0)[batch]
            # else:
            #     # 如果是单一图，就直接平均所有节点
            #     graph_context = torch.mean(x, dim=0, keepdim=True).expand_as(x)
            #
            # # 编码上下文
            # #encoded_context = self.context_encoder(graph_context)
            #
            # in_logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
            #                             act_edge_attr=act_edge_embedding)  # (N, 2)
            # out_logits = self.out_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
            #                               act_edge_attr=act_edge_embedding)  # (N, 2)
            # # print("x shape:", x.shape)
            # # print("in_logits shape:", in_logits.shape)
            #
            #
            # feature_diff = x - graph_context
            # difference_aware = torch.cat([x, feature_diff], dim=-1)
            # refined_in_logits = in_logits + self.action_refiner(difference_aware)[:, :2]
            # refined_out_logits = out_logits + self.action_refiner(difference_aware)[:, 2:]
            # #print("feature_diff shape:", feature_diff.shape)
            # # refined_outputs = self.action_refiner(feature_diff)
            # # refined_in_logits = in_logits + refined_outputs[:, :2]
            # # refined_out_logits = out_logits + refined_outputs[:, 2:]
            # #print("refine shape:", refined_in_logits.shape)
            #
            # temp = self.temp_model(x=x, edge_index=edge_index,
            #                       edge_attr=env_edge_embedding) if self.learn_temp else self.temp
            # in_probs = F.gumbel_softmax(logits=refined_in_logits, tau=temp, hard=True)
            # out_probs = F.gumbel_softmax(logits=refined_out_logits, tau=temp, hard=True)
            ##############################################
            #历史决策信息引入——1
            # 从决策历史生成条件参数

            #############################################
            # 当前层的历史信息切片
            current_history = self.action_history.clone()


            # action
            in_logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                        act_edge_attr=act_edge_embedding,history=current_history)  # (N, 2)
            out_logits = self.out_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                          act_edge_attr=act_edge_embedding, history=current_history)  # (N, 2)

            temp = self.temp_model(x=x, edge_index=edge_index,
                                   edge_attr=env_edge_embedding) if self.learn_temp else self.temp
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            edge_weight = self.create_edge_weight(edge_index=edge_index,
                                                  keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])

            # 创建当前层的动作记录
            current_actions = torch.zeros(num_nodes, 4, device=x.device)
            current_actions[:, 0] = in_probs[:, 0] * out_probs[:, 0]  # S
            current_actions[:, 1] = in_probs[:, 0] * (1 - out_probs[:, 0])  # L
            current_actions[:, 2] = (1 - in_probs[:, 0]) * out_probs[:, 0]  # B
            current_actions[:, 3] = (1 - in_probs[:, 0]) * (1 - out_probs[:, 0])  # I

            # 更新历史记录
            if gnn_idx < self.num_layers - 1:  # 最后一层不需要更新历史
                updated_history = updated_history.roll(-4, dims=1)
                updated_history[:, -4:] = current_actions

            # environment
            out = self.env_net[1 + gnn_idx](x=x, edge_index=edge_index, edge_weight=edge_weight,
                                            edge_attr=env_edge_embedding)
            out = self.dropout(out)
            out = self.act(out)

            if calc_stats:
                edge_ratio = edge_weight[edge_ratio_edge_mask].sum() / edge_weight[edge_ratio_edge_mask].shape[0]
                edge_ratio_list.append(edge_ratio.item())

            if self.skip:
                x = x + out
            else:
                x = out

        x = self.hidden_layer_norm(x)
        x = self.pooling(x, batch=batch)
        x = self.env_net[-1](x)  # decoder
        result = result + x

        if calc_stats:
            edge_ratio_tensor = torch.tensor(edge_ratio_list, device=x.device)
        else:
            edge_ratio_tensor = -1 * torch.ones(size=(self.num_layers,), device=x.device)
        return result, edge_ratio_tensor, updated_history

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob
