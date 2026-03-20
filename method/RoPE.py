import torch

"""
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([10,20,30,40,50])

condition = x>3

result = torch.where(condition,x,y)

print(result)
"""

"""
t1 = torch.range(0,10,2)
print(t1)

t2 = torch.range(5,0,-1)
print(t2)
"""

"""
v1 = torch.tensor([1,2,3])
v2 = torch.tensor([4,5,6])
result = torch.outer(v1,v2)
print(result)
"""

"""
t1 = torch.tensor([[[1,2,3],[4,5,6]],[[13,14,15],[16,17,18]]])
t2 = torch.tensor([[[7,8,9],[10,11,12]],[[19,20,21],[22,23,24]]])
print(t1.shape)#(2,2,3)
result1 = torch.cat((t1,t2),dim=0)
print(result1)#(4,2,3)
result2 = torch.cat((t1,t2),dim=1)
print(result2)#(2,4,3)
"""

t1 = torch.tensor([1,2,3])
t2 = t1.unsqueeze(0)
print(t1.shape)
print(t2)
print(t2.shape)