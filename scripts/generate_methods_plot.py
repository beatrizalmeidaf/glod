import matplotlib.pyplot as plt
import numpy as np

# Dark mode styling
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#0a0a10')
ax.set_facecolor('#0a0a10')

methods = ['GPTQ', 'AWQ', 'Wanda', 'RTN', 'SparseGPT']
offsets = [-0.00148, -0.00146, 0.00372, 0.00368, 0.01182]
se = [0.00179, 0.00193, 0.00335, 0.00096, 0.00321]

# Convert to percentage points for better readability on chart
offsets_pct = [x * 100 for x in offsets]
se_pct = [x * 100 for x in se]

x = np.arange(len(methods))
colors = ['#0055ff', '#0055ff', '#7000ff', '#ffffff', '#7000ff']

# Bar chart with error bars
rects = ax.bar(x, offsets_pct, yerr=se_pct, capsize=6, color=colors, alpha=0.8, edgecolor='none', error_kw=dict(ecolor='#a0a0b0', lw=2, capsize=6, capthick=2))

ax.set_ylabel('Desvio vs GLOD Teórico (% de Flips)', color='#a0a0b0', fontsize=12)
ax.set_title('Aderência Universal: Métodos vs Lei Geométrica (GLOD)', color='white', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, color='white', fontsize=11, fontweight='bold')
ax.axhline(y=0.0, color='#00f0ff', linestyle='-', linewidth=2, label='Fórmula GLOD (Erro Zero)')

ax.legend(facecolor='#15151a', edgecolor='#333333', framealpha=0.9, loc='upper left')

# Add values on top/bottom of bars
for rect, offset in zip(rects, offsets_pct):
    height = rect.get_height()
    y_pos = height + 0.15 if height > 0 else height - 0.25
    ax.text(rect.get_x() + rect.get_width() / 2, y_pos,
            f'{offset:+.2f}pp',
            ha='center', va='bottom' if height > 0 else 'top', color='white', fontsize=10, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333')
ax.spines['bottom'].set_color('#333')
ax.grid(axis='y', color='#ffffff', alpha=0.05, linestyle='--')

plt.tight_layout()
plt.savefig('results/graficos/methods_chart.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
