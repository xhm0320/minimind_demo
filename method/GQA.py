import torch
import torch.nn as nn

dropout_layer = nn.Dropout(p=0.5)

t1 = torch.Tensor([1,2,3])
t2 = dropout_layer(t1)

# print(t2)

# layer = nn.Linear(in_features = 3,out_features = 5,bias = True)

# t1 = torch.Tensor([1,2,3])
# t2 = torch.Tensor([[1,2,3]])

# output2 = layer(t2)
# print(output2)

# t = torch.Tensor([[1,2,3,4,5,6],[7,8,9,10,11,12]])#[2,6]
# t_view1 = t.view(3,4)
# print(t_view1)
# t_view2 = t.view(4,3)
# print(t_view2)

# x = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(torch.triu(x))

# print(torch.triu(x,diagonal=-1))