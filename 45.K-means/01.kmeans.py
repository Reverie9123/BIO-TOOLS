import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

file_path = 'QianJinTeng.count.txt.tpm'
data = pd.read_csv(file_path, delimiter='\t', index_col=0)
data = data.loc[(data != 0).any(axis=1)]

data_mean = pd.DataFrame()
data_mean['fruit'] = data[['fruit1', 'fruit2']].mean(axis=1)
data_mean['leaf'] = data[['leaf1', 'leaf2']].mean(axis=1)
data_mean['root'] = data[['root1', 'root2']].mean(axis=1)
data_mean['stem'] = data[['stem1', 'stem2']].mean(axis=1)
log_data_mean = np.log2(data_mean + 1)

data_transposed = data_mean.T
scaler = StandardScaler()
scaled_data_mean = scaler.fit_transform(data_transposed)

scaled_data_mean=scaled_data_mean.T

n_clusters = 4
kmeans_mean = KMeans(n_clusters=n_clusters, random_state=0)
clusters_mean = kmeans_mean.fit_predict(scaled_data_mean)

clustered_data_mean = data_mean.copy()
clustered_data_mean['Cluster'] = clusters_mean

output_file_path = 'clustered_data.csv'
clustered_data_mean.to_csv(output_file_path, sep='\t', index=True)

scaled_data_mean_df = pd.DataFrame(scaled_data_mean, index=data_mean.index, columns=data_mean.columns)
scaled_data_mean_df['Cluster'] = clusters_mean
# scaled_data_mean_df.to_csv("scaled_data_mean.txt", sep='\t', index=True)

rows = 2  
cols = 2  

line_color = 'orange' 
line_alpha = 0.01  
x_labels = ["fruit", "root", "leaf", "stem"]

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

with PdfPages('kmeans.pdf') as pdf:
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    for cluster in range(n_clusters):
        ax = axes[cluster // cols, cluster % cols]
        cluster_data = scaled_data_mean_df[scaled_data_mean_df['Cluster'] == cluster].drop('Cluster', axis=1)
        cluster_data = cluster_data.reindex(columns=x_labels)
        
        for index, row in cluster_data.iterrows():
            ax.plot(row, color=line_color, alpha=line_alpha)
        ax.plot(cluster_data.mean(), color='black')
        ax.set_title(f'Cluster {cluster}')
        # ax.set_xlabel('Organ')
        ax.set_ylabel('Scaled gene expression quantity')
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)
