from statistics import mean, stdev
import numpy as np

def normalize(X: np.ndarray, offset: int) -> np.ndarray:
    for i in range(offset, X.shape[1]):
        col = X[:, i]
        m = mean(col.tolist())
        s = stdev(col.tolist())
        col -= m
        col /= s
    return X

def relativize(X: np.ndarray, offset: int) -> np.ndarray:
    result = np.empty((X.shape[0], offset + 16), dtype=X.dtype)
    for a, b in zip(X, result):
        b[:offset] = a[:offset]
        x = a[offset::2]
        y = a[offset+1::2]
        # Reference points
        #mline = mean((y[61], y[62], y[63]))
        #mline2 = mean((y[65], y[66], y[67]))
        # Custom features
        # Vertical
        b[offset] = mean((y[27], y[28], y[29], y[30]))
        b[offset+1] = mean((y[31], y[32], y[34], y[35]))
        b[offset+2] = mean((abs(y[38]-y[40]), abs(y[43]-y[47])))
        b[offset+3] = mean((y[49], y[63]))
        b[offset+4] = mean((y[0], y[16]))
        b[offset+5] = mean((y[21], y[22]))
        b[offset+6] = mean((y[60], y[64]))
        b[offset+7] = mean((y[55], y[59]))
        b[offset+8] = mean((y[56], y[57], y[58]))
        # Horizontal
        b[offset+9] = abs(x[39]-x[42])
        b[offset+10] = abs(mean((x[31], x[32])) - mean((x[34], x[35])))
        b[offset+11] = abs(x[38]-x[43])
        b[offset+12] = abs(x[21]-x[22])
        b[offset+13] = abs(x[18]-x[25])
        b[offset+14] = abs(x[49]-x[53])
        b[offset+15] = abs(x[5]-x[11])
    return result
