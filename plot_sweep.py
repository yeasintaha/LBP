import json
import matplotlib.pyplot as plt
import os

with open('results/sweep_orl.json') as f:
    data = json.load(f)
    
results = data['results']


plt.figure(figsize=(10, 6))

configs = ['LBP_8_1', 'LBP_8_2', 'LBP_16_2']
labels = ['LBP(8,1)', 'LBP(8,2)', 'LBP(16,2)']
markers = ['o', 's', '^']

for config, label, marker in zip(configs, labels, markers):
    x_vals = []
    y_vals = []
    
    keys = [k for k in results.keys() if k.startswith(config)]
    keys.sort(key=lambda x: results[x]['k'])
    
    for key in keys:
        k = results[key]['k']
        x_vals.append(k * k)
        y_vals.append(results[key]['mean_rate']['chi2'] * 100)
        
    plt.plot(x_vals, y_vals, marker=marker, linestyle='-', label=label, markersize=8)

plt.xlabel('Number of Regions')
plt.ylabel('Recognition Rate (%)')
plt.title('Performance on ORL database (Chi-Square Distance)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

output_path = 'results/figure_4_reproduction.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph saved to: {output_path}")
