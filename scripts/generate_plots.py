import matplotlib.pyplot as plt
import numpy as np

# Dark mode styling
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0a0a10')
ax.set_facecolor('#0a0a10')

corpora = ['mix (paper)', 'gsm8k', 'mmlu_en', 'wikitext']
maligno = [1.13, 1.17, 1.00, 1.04]
benigno = [0.51, 0.55, 0.56, 0.59]

x = np.arange(len(corpora))
width = 0.35

rects1 = ax.bar(x - width/2, maligno, width, label='Maligno (Max Flips)', color='#ff5f56')
rects2 = ax.bar(x + width/2, benigno, width, label='Benigno (Min Flips)', color='#00f0ff')

ax.set_ylabel('Razão (κ / κ_honesto)', color='#a0a0b0', fontsize=12)
ax.set_title('Ataques Adversariais vs Compressores Honestos por Corpus', color='white', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(corpora, color='#a0a0b0')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline Honesto (1.0x)')

ax.legend(facecolor='#15151a', edgecolor='#ffffff', framealpha=0.8)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}x',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='white', fontsize=10)

autolabel(rects1)
autolabel(rects2)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333')
ax.spines['bottom'].set_color('#333')

plt.tight_layout()
plt.savefig('results/graficos/attacks_chart.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
