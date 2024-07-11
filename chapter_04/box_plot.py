import numpy as np
import matplotlib.pyplot as plt

d = [
    [0.6930, -1.1259, -1.5318, 0.9525, 1.1824],
    [0.5464, -0.0120, 0.5051, -0.0192, -0.1141],
    [0.8912, 1.3826, 1.5193, -1.1996, -1.1403],
    [1.1690, 0.4970, -0.1712, -0.5340, 0.3047],
    [-0.9221, -0.1071, 0.3079, -0.3885, -0.4753],
    [1.5699, -1.4767, 0.3327, 1.4714, 1.1807],
    [-0.3479, 0.4775, 1.8823, -1.4031, -0.7396],
    [0.0887, -0.4353, -1.7377, -1.2349, 1.7456],
    [1.0775, 0.9524, 1.2475, 0.7291, -1.1207],
    [-1.4657, 0.9250, -1.0446, 0.4262, -1.0279],
    [-1.3332, 1.4501, 0.0323, 1.1102, -0.8966],
    [0.3005, -1.4500, -0.2615, 1.7033, -0.2505],
    [-1.4377, -0.2472, -0.4340, -0.7032, 0.3362],
    [0.3016, -1.5527, -0.6213, 0.1780, -0.7517],
    [-1.1315, 0.7225, -0.0250, -1.0881, 1.7674],
]
d = np.array(d)
plt.boxplot(d)
plt.show()
