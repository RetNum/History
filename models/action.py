import torch.nn as nn
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch import chunk

from helpers.classes import ActionNetArgs


class ActionNet(nn.Module):
    ##添加history_dim参数
    def __init__(self, action_args: ActionNetArgs, history_dim: int = None):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        self.num_layers = action_args.num_layers
        self.net = action_args.load_net()
        self.dropout = nn.Dropout(action_args.dropout)
        self.act = action_args.act_type.get()

        if history_dim is not None:
            self.use_history = True
            self.condition_net = nn.Linear(history_dim, action_args.env_dim * 2)
        else:
            self.use_history = False

        # 添加历史条件处理模块
        #self.condition_net = nn.Linear(history_dim, action_args.hidden_dim * 2)

    def forward(self, x: Tensor, edge_index: Adj, env_edge_attr: OptTensor,
                act_edge_attr: OptTensor, history: OptTensor = None) -> Tensor:

        # 条件调制节点特征
        if history is not None:
            # 生成调制参数
            condition_params = self.condition_net(history)
            gamma, beta = chunk(condition_params, 2, dim=1)

            # 应用调制
            x = gamma * x + beta

        # 原有的前向传播逻辑
        edge_attrs = [env_edge_attr] + (self.num_layers - 1) * [act_edge_attr]
        for idx, (edge_attr, layer) in enumerate(zip(edge_attrs[:-1], self.net[:-1])):
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.dropout(x)
            x = self.act(x)
        x = self.net[-1](x=x, edge_index=edge_index, edge_attr=edge_attrs[-1])

        return x



    # def forward(self, x: Tensor, edge_index: Adj, env_edge_attr: OptTensor, act_edge_attr: OptTensor) -> Tensor:
    #     edge_attrs = [env_edge_attr] + (self.num_layers - 1) * [act_edge_attr]
    #     for idx, (edge_attr, layer) in enumerate(zip(edge_attrs[:-1], self.net[:-1])):
    #         x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
    #         x = self.dropout(x)
    #         x = self.act(x)
    #     x = self.net[-1](x=x, edge_index=edge_index, edge_attr=edge_attrs[-1])
    #     return x
