import json
import csv
import matplotlib.pyplot as plt
import numpy as np

with open('results/orl_seed0.json') as f:
    orl_data = json.load(f)['results']

rates = orl_data['rates']

plt.figure(figsize=(8, 5))
plt.hist(rates, bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Recognition Rates over 100 Permutations (ORL)')
plt.xlabel('Recognition Rate')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('results/figure_orl_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: results/figure_orl_histogram.png")

with open('results/sweep_orl.json') as f:
    sweep_data = json.load(f)['results']

with open('results/table_1_sweep.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['LBP Configuration', 'Grid (k)', 'Window Size', 'Feature Length', 'Chi-Square Acc', 'HI Acc', 'LL Acc'])
    
    for key in sorted(sweep_data.keys()):
        row = sweep_data[key]
        lbp_name = key.split('|')[0].replace('_', '(').replace('(', '(P=', 1).replace('LBP', 'LBP') # formatting
        writer.writerow([
            key.split('|')[0],
            row['k'],
            f"{row['window'][0]}x{row['window'][1]}",
            row['feature_length'],
            f"{row['mean_rate']['chi2']:.4f}",
            f"{row['mean_rate']['intersection']:.4f}",
            f"{row['mean_rate']['log_likelihood']:.4f}"
        ])
print("Saved: results/table_1_sweep.csv")

with open('results/feret_demo.json') as f:
    feret_data = json.load(f)['results']

probes = ['fb', 'fc', 'dup1', 'dup2']
weighted = [feret_data['rank1'][p]['weighted'] for p in probes]
nonweighted = [feret_data['rank1'][p]['nonweighted'] for p in probes]


with open('results/table_feret.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Probe Set', 'Weighted LBP Acc', 'Non-weighted LBP Acc'])
    for p, w, nw in zip(probes, weighted, nonweighted):
        writer.writerow([p, f"{w:.4f}", f"{nw:.4f}"])
print("Saved: results/table_feret.csv")


x = np.arange(len(probes))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, weighted, width, label='Weighted LBP', color='#4CAF50')
rects2 = ax.bar(x + width/2, nonweighted, width, label='Non-weighted LBP', color='#2196F3')

ax.set_ylabel('Recognition Rate')
ax.set_title('FERET Synthetic Data: Weighted vs Non-weighted (Rank 1)')
ax.set_xticks(x)
ax.set_xticklabels(probes)
ax.set_ylim([0, 1.1])
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

plt.tight_layout()
plt.savefig('results/figure_feret_bar.png', dpi=300)
plt.close()
print("Saved: results/figure_feret_bar.png")

