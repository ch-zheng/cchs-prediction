import math
import numpy as np

betas = np.load('test.npy').squeeze()
named_betas = []
for i, b in enumerate(betas):
    if i == 0:
        p = 'race'
    elif i == 1:
        p = 'age'
    else:
        p = 'x' if i % 2 == 0 else 'y'
    coord = math.floor((i-2)/2)
    named_betas.append((p + str(coord), b))
named_betas = sorted(named_betas, key=lambda x: -abs(x[1]))
print(named_betas)
