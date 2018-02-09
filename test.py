import torch
from torch.autograd import Variable
x = Variable(torch.ones(2,2),requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y*y + 5
print(z)

out = z.mean()
print(out)

out.backward()
print(x.grad)
