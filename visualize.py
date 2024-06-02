import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


matric = {
    "noisy": {
        "pesq_nb": 2.5718609927350076,
        "pesq_wb": 1.9527549052046491,
        "rtf": 5.301575962285659e-07,
        "stoi": 0.9275962661988458
    },
    "spectral_subtraction": {
        "pesq_nb": 2.7653248250646167,
        "pesq_wb": 2.14419878083159,
        "rtf": 0.040543086745086866,
        "stoi": 0.9242903495636744
    }
}


labels = ['pesq_nb', 'pesq_wb', 'rtf', 'stoi']


noisy_data = [matric['noisy'][label] for label in labels]


spectral_data = [matric['spectral_subtraction'][label] for label in labels]


x = np.arange(len(labels))

width = 0.35


fig, ax = plt.subplots()


rects1 = ax.bar(x - width/2, noisy_data, width, label='noisy')
rects2 = ax.bar(x + width/2, spectral_data, width, label='spectral_subtraction')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Values')
ax.set_title('Comparison of Noisy and Spectral Subtraction')
ax.legend()

fig.tight_layout()


plt.show()
