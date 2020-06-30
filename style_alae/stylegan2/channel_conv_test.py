import torch

import util

inp = torch.cat([torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1)], dim=0)

weights = torch.ones(1, 1, 1, 1, 1)
style = torch.cat([torch.ones(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1)], dim=0)
weights = style * weights
weights = util.apply_weight_norm(weights, input_dims=(1, 3, 4))
weights = weights.view(2, 1, 1, 1)
x = inp.view(1, 2, 1, 1)
outp = torch.nn.functional.conv_transpose2d(x, weights, None, 1, 0, 0, 2)
outp = outp.view(2, 1, 1, 1)
print(outp)
