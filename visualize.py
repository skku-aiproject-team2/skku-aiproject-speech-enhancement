import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

# x 축 레이블
labels = ['pesq_nb', 'pesq_wb', 'rtf', 'stoi']

# 각 방법의 데이터를 리스트로 정리
noisy_data = [metric['noisy'][label] for label in labels]
spectral_data = [metric['spectral_subtraction'][label] for label in labels]
mmse_data = [metric['mmse'][label] for label in labels]
wiener_data = [metric['wiener_filtering'][label] for label in labels]


x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 8))

# 각 데이터 방법에 대한 막대 그래프
rects1 = ax.bar(x - 1.5*width, noisy_data, width, label='noisy')
rects2 = ax.bar(x - 0.5*width, spectral_data, width, label='spectral_subtraction')
rects3 = ax.bar(x + 0.5*width, mmse_data, width, label='mmse')
rects4 = ax.bar(x + 1.5*width, wiener_data, width, label='wiener_filtering')

# x축 레이블 
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Values')
ax.set_title('Comparison of Different Denoising Methods')
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)


fig.tight_layout()
plt.show()
