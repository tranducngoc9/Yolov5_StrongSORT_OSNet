import numpy as np
filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
filter_du = np.stack([filter_du] * 3, axis=2)

print(filter_du)