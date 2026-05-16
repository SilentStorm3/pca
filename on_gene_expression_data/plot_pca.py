import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
df_class = pd.read_csv('data/class.tsv', header=None, names=['label'])
df_filtered = pd.read_csv('data/filtered.tsv.gz', sep='\t')
df_filtered.columns = df_filtered.columns.str.strip()

labels = df_class['label'].values
colors = np.array(['red' if label == 1 else 'black' for label in labels])

gata3 = df_filtered['4359'].values
xbp1 = df_filtered['4404'].values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel a: Raw expression
axes[0].scatter(gata3, xbp1, c=colors, alpha=0.7, edgecolor='k')
axes[0].set_xlabel('GATA3')
axes[0].set_ylabel('XBP1')
axes[0].set_title('Raw Expression')

# Panel c: Projection onto PC1
X = np.column_stack((gata3, xbp1))
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pc1_scores = pca.fit_transform(X_scaled)[:, 0]

er_minus = (labels == 0)
er_plus = (labels == 1)

axes[1].scatter(pc1_scores, np.full_like(pc1_scores, 2), c=colors, alpha=0.7, edgecolor='k')
axes[1].scatter(pc1_scores[er_minus], np.full_like(pc1_scores[er_minus], 1), c='black', alpha=0.7, edgecolor='k')
axes[1].scatter(pc1_scores[er_plus], np.zeros_like(pc1_scores[er_plus]), c='red', alpha=0.7, edgecolor='k')

axes[1].set_yticks([0, 1, 2])
axes[1].set_yticklabels(['ER+', 'ER-', 'All'])
axes[1].set_xlabel('Score on PC 1')
axes[1].set_title('Projection onto PC1')

plt.tight_layout()
plt.savefig('pca.png', dpi=300)
print("Saved pca.png")
