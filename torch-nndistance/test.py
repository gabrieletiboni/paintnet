import torch
import torch_nndistance as NND

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
points1 = torch.rand(16, 2048, 3).to(device)
points1.requires_grad = True
points2 = torch.rand(16, 1024, 3).to(device)
points2.requires_grad = True

dist1, dist2, idx1, idx2 = NND.nnd(points1, points2)
# print(dist1, dist2)
print('dist1: ', dist1.size())
print('dist2: ', dist2.size())

loss = torch.sum(dist1)
print(loss)
loss.backward()
print(points1.grad, points2.grad)
print('ok - Test 1')
print("points1 grad:\n", points1.grad)
print("")
print("points2 grad:\n", points2.grad)

if points1.grad is not None and points2.grad is not None:
    print('ok - Test 2 Gradient')
else:
    print('Fail - Test 2 Gradient')
