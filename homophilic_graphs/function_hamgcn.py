import torch
from torch import nn
import torch_sparse
import torch.nn.functional as F
from base_classes import ODEFunc
from utils import MaxNFEException
from torch_geometric.nn.conv import GCNConv
from utils import get_rw_adj
import numpy as np
# def batch_jacobian(func, x, create_graph=False):
#   # x in shape (Batch, Length)
#   def _func_sum(x):
#     return func(x).sum(dim=0)        ###readout, pooling, mean, square, norm2, asb,l1 norm
#
#   return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 2, 0)

def batch_jacobian(func, x, create_graph=False):
  # x in shape (Batch, Length)
  # def _func_sum(x):
  #   return func(x).sum(dim=0)        ###readout, pooling, mean, square, norm2, asb,l1 norm

  return torch.autograd.functional.jacobian(func, x, create_graph=create_graph).permute(1, 2, 0)


class attention_H(nn.Module):
  """"replace this module by a aggregation function """

  def __init__(self, size_in, edge_index):
    super().__init__()
    self.dim = size_in

    self.layer1 =GCNConv(size_in*2, size_in, normalize=True)
    self.edge_index = edge_index
    self.layer2 =GCNConv(size_in,1, normalize=True)
    self.dropout = nn.Dropout(p=0.4)
  def forward(self, x):
    #
    out = self.layer1(x,self.edge_index)
    out = torch.tanh(out)
    # out = torch.relu(out)
    # out = self.dropout(out)
    out = self.layer2(out,self.edge_index)
    # out = torch.tanh(out)
    out = torch.norm(out, dim=0)
    # print("out.shape: ", out.shape)
    return out

class H_x(nn.Module):
  """"replace this module by a aggregation function """

  def __init__(self, size_in, edge_index):
    super().__init__()
    self.dim = size_in

    self.layer1 =GCNConv(size_in, size_in, normalize=True)
    self.edge_index = edge_index
    self.layer2 =GCNConv(size_in,1, normalize=True)

  def forward(self, x):
    #
    out = self.layer1(x,self.edge_index)
    out = torch.tanh(out)
    # out = torch.relu(out)
    out = self.layer2(out,self.edge_index)
    out = torch.norm(out, dim=0)
    return out

class H_x_linear(nn.Module):
  """"replace this module by a aggregation function """

  def __init__(self, size_in, edge_index):
    super().__init__()
    self.dim = size_in

    self.layer1 =nn.Linear(size_in, size_in,)
    self.edge_index = edge_index
    self.layer2 =nn.Linear(size_in,1,)

  def forward(self, x):
    #
    out = self.layer1(x,)
    out = torch.tanh(out)
    # out = torch.relu(out)
    out = self.layer2(out,)
    out = torch.norm(out, dim=0)
    return out

class H_derivatie(nn.Module):
  """"replace this module by a aggregation function """

  def __init__(self, size_in, edge_index):
    super().__init__()
    self.dim = size_in

    self.layer1 = GCNConv(size_in * 2, size_in*2, normalize=True)
    self.edge_index = edge_index
    self.layer2 = GCNConv(size_in*2, size_in*2, normalize=True)

  def forward(self, x):
    #
    # print('x in H: ',x.type())
    # print('edge_index in H: ', self.edge_index.type())
    out = self.layer1(x,self.edge_index)
    out = torch.tanh(out)
    out = self.layer2(out,self.edge_index)
    return out

class H_derivatie_x(nn.Module):
  """"replace this module by a aggregation function """

  def __init__(self, size_in, edge_index):
    super().__init__()
    self.dim = size_in

    self.layer1 = GCNConv(size_in, size_in, normalize=True)
    self.edge_index = edge_index
    self.layer2 = GCNConv(size_in, size_in, normalize=True)

  def forward(self, x):
    #
    # print('x in H: ',x.type())
    # print('edge_index in H: ', self.edge_index.type())
    out = self.layer1(x,self.edge_index)
    out = torch.tanh(out)
    # out = torch.sin(out)
    out = self.layer2(out,self.edge_index)
    # out = torch.sin(out)
    return out


class H_derivatie_x_linear(nn.Module):
  """"replace this module by a aggregation function """

  def __init__(self, size_in, edge_index):
    super().__init__()
    self.dim = size_in

    self.layer1 = nn.Linear(size_in, size_in, )
    self.edge_index = edge_index
    self.layer2 = nn.Linear(size_in, size_in, )

  def forward(self, x):
    #
    # print('x in H: ',x.type())
    # print('edge_index in H: ', self.edge_index.type())
    out = self.layer1(x,)
    out = torch.tanh(out)
    # out = torch.sin(out)
    out = self.layer2(out,)
    # out = torch.sin(out)
    return out

class linear_H(nn.Module):
  """"replace this module by a aggregation function """

  def __init__(self, size_in, edge_index, edge_weight):
    super().__init__()
    self.dim = size_in

    self.layer1 = nn.Linear(size_in*2, size_in )
    self.edge_index = edge_index
    self.layer2 = nn.Linear(size_in, 1, )
    self.edge_weight = edge_weight

  def forward(self, x):
    #
    # print('x in H: ',x.type())
    # print('edge_index in H: ', self.edge_index.type())
    out = self.layer1(x, )
    out = torch.sin(out)
    out = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], out)
    out = self.layer2(out,)
    out = torch.sin(out)
    # out = x

    out = torch.norm(out, dim=0)
    # out = torch.reshape(out,shape=[out1.shape[0],out1.shape[1]])
    # out = out.sum(dim=0)
    
    # out = self.layer2(out,)
    # print("out before spmm: ", out.shape)

    # print("out after spmm: ", out.shape)
    return out

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class HAMGCNFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(HAMGCNFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))
    self.conv = GCNConv(in_features, out_features, normalize=True)

    self.edge_index, self.edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                         fill_value=opt['self_loop_weight'],
                                         num_nodes=data.num_nodes,
                                         dtype=data.x.dtype)

    # if self.training:
    #     print("drop edge here")
    #     self.dropedge_perc = .5
    #     if self.dropedge_perc < 1:
    #         nnz = len(self.edge_weight)
    #         perm = np.random.permutation(nnz)
    #         preserve_nnz = int(nnz*self.dropedge_perc)
    #         perm = perm[:preserve_nnz]
    #         # self.odefunc.edge_attr = self.odefunc.edge_attr[perm]
    #         self.edge_index =self.edge_index[:,perm]
    #         self.edge_weight = self.edge_weight[perm]
    # else:
    #     self.dropedge_perc = 1
    #     if self.dropedge_perc < 1:
    #         nnz = len(self.edge_weight)
    #         perm = np.random.permutation(nnz)
    #         preserve_nnz = int(nnz*self.dropedge_perc)
    #         perm = perm[:preserve_nnz]
    #         # self.odefunc.edge_attr = self.odefunc.edge_attr[perm]
    #         self.edge_index = self.edge_index[:,perm]
    #         self.edge_weight = self.edge_weight[perm]
    self.conv_v = GCNConv(in_features, out_features, normalize=True)
    self.conv_x = GCNConv(in_features, out_features, normalize=True)
    self.conv_full = GCNConv(int(in_features*2), int(in_features*2), normalize=True)
    # self.H = myLinear(in_features)
    self.edge_index =self.edge_index.to(device)
    self.edge_weight = self.edge_weight.to(device)
    # self.H = attention_H(in_features,self.edge_index)
    # self.H_1 = H_x(in_features, self.edge_index)
    # self.H_2 = H_x(in_features, self.edge_index)
    # self.H = linear_H(in_features, self.edge_index,self.edge_weight)
    # self.H = H_derivatie(in_features, self.edge_index)

    self.H_derivatie_x= H_derivatie_x(in_features, self.edge_index)
    self.H_x = H_x(in_features, self.edge_index)
    #
    # self.H_derivatie_x = H_derivatie_x_linear(in_features, self.edge_index)
    # self.H_x = H_x_linear(in_features, self.edge_index)
  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      # ax = torch.mean(torch.stack(
      #   [torch_sparse.spmm(self.edge_index, self.attention_weights[:, idx], x.shape[0], x.shape[0], x) for idx in
      #    range(self.opt['heads'])], dim=0), dim=0)
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x_full):  # the t param is needed by the ODE solver.
    x = x_full[:, :self.opt['hidden_dim']]
    y = x_full[:, self.opt['hidden_dim']:]

    # H_derivatie = batch_jacobian(lambda xx: self.H(xx), x_full, create_graph=True).squeeze()
    #
    # dx = H_derivatie[:,self.opt['hidden_dim']:]
    # dv = -1 * H_derivatie[:, :self.opt['hidden_dim']]
    #
    # f = torch.hstack([dx, dv])

    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    ay = self.sparse_multiply(y)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    # f = self.conv(y, self.edge_index) - x - y
    # print('edge_index in HamGNN: ', self.edge_index.type())
    # f_v = self.conv_v(x,self.edge_index) - x
    # # f_x = self.conv_x(y,self.edge_index) - y
    # f_x = -1 * batch_jacobian(lambda xx: self.H(xx), y, create_graph=True).squeeze()
    # f = torch.hstack([f_v, f_x])

    # f = torch.hstack([f_x, f_v])

    # f = self.conv_full(x_full,self.edge_index)
    # f_v = -1 *batch_jacobian(lambda xx: self.H_1(xx), x, create_graph=True).squeeze()
    # f_x =  batch_jacobian(lambda xx: self.H_2(xx), y, create_graph=True).squeeze()
    # f = torch.hstack([f_x, f_v])


    # f_full = batch_jacobian(lambda xx: self.H(xx), x_full, create_graph=True).squeeze()
    # # f_full = self.H(x_full)
    # dx = f_full[..., self.in_features:]
    # dv = -1 * f_full[..., 0:self.in_features]
    # f_v = dv
    # f = torch.hstack([dx, dv])


    # f_x = self.H_derivatie_x(y)
    # f_v = -1 *batch_jacobian(lambda xx: self.H_x(xx), x, create_graph=True).squeeze()

    f_v = self.H_derivatie_x(x)
    f_x = -1 * batch_jacobian(lambda xx: self.H_x(xx), y, create_graph=True).squeeze()
    f = torch.hstack([f_x, f_v])

    # if not self.training:
    #   energy = self.H_x(y)
    #   print("t: ", t)
    #   print("energy: ",energy)

    # f_x = batch_jacobian(lambda xx: self.H(xx), x, create_graph=True).squeeze()
    # dv =f_x
    # f_v = dv -y-x
    # f_x = dx
    # f = torch.hstack([dx, dv])
    # f = (ay - x - y)
    # if self.opt['add_source']:
    #   f = (1. - F.sigmoid(self.beta_train)) * f + F.sigmoid(self.beta_train) * self.x0[:, self.opt['hidden_dim']:]
    # f = torch.cat([f_v, (1. - F.sigmoid(self.beta_train2)) * alpha * x + F.sigmoid(self.beta_train2) *
    #                self.x0[:, :self.opt['hidden_dim']]], dim=1)
    return f
