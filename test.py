import zoo
from pathlib import Path
import numpy as np

model = zoo.Logistic()
model.load(Path('pretrained', 'logistic' + '.pickle'))
np.save('test.npy', model.model.coef_)
