import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

# 1. 데이터베이스 로드
conn = sqlite3.connect('papers.db')

df = pd.read_sql_query("SELECT title, journal FROM papers", conn)
conn.close()

sample_title = "Machine-learning potential for silver sulfide: From CHGNet pretraining to DFT-refined phase stability"

print(f"총 {len(df)}개의 논문을 분석합니다...")

# 2. SPECTER 2 모델로 벡터 변환 (무료/로컬 실행)
model = SentenceTransformer('allenai/specter2_base')

# DB 논문 + sample_title 을 함께 임베딩
all_titles = df['title'].tolist() + [sample_title]
all_embeddings = model.encode(all_titles, show_progress_bar=True)

paper_embeddings = all_embeddings[:-1]   # DB 논문들
sample_embedding = all_embeddings[-1:]   # sample 1개

# 3. KMeans 클러스터링 (k=5)
N_CLUSTERS = 5
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(paper_embeddings)
sample_cluster = kmeans.predict(sample_embedding)[0]

# 4. t-SNE 3D 압축 (sample 포함하여 함께 변환)
tsne = TSNE(n_components=3, random_state=42)
all_3d = tsne.fit_transform(all_embeddings)

embeddings_3d = all_3d[:-1]      # DB 논문들의 3D 좌표
sample_3d = all_3d[-1]           # sample의 3D 좌표

# 5. 시각화 - 3D 클러스터별 색상 + sample 빨간점 + 클릭 정보
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# 클러스터별 색상 팔레트
cmap = plt.colormaps['tab10']
colors = [cmap(c) for c in df['cluster']]

# DB 논문 scatter (클러스터 색상)
scatter = ax.scatter(
    embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
    c=df['cluster'], cmap='tab10', alpha=0.6, s=40,
    edgecolors='white', linewidths=0.3,
)

# sample_title 빨간 점 (크게, 별 모양)
ax.scatter(
    sample_3d[0], sample_3d[1], sample_3d[2],
    c='red', s=200, marker='*', edgecolors='black', linewidths=0.8,
    zorder=5, label=f"Sample (Cluster {sample_cluster})",
)

# 범례: 클러스터 + sample
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10,
               label=f'Cluster {i}') for i in range(N_CLUSTERS)
]
handles.append(
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markeredgecolor='black', markersize=15, label='Sample Paper')
)
ax.legend(handles=handles, loc='best', fontsize=9)

ax.set_title("Paper Semantic Map (SPECTER 2 + t-SNE 3D, K=5 Clusters)")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")

# 클릭 시 가장 가까운 논문 정보 표시
annot = ax.text2D(0.02, 0.98, "", transform=ax.transAxes,
                  fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", alpha=0.95))
annot.set_visible(False)


def on_click(event):
    """마우스 클릭 시 가장 가까운 논문의 제목, 저널, 클러스터 정보를 표시"""
    if event.inaxes != ax:
        return

    # 3D 투영 좌표 기반으로 가장 가까운 점 찾기
    from mpl_toolkits.mplot3d import proj3d
    click_x, click_y = event.xdata, event.ydata

    # 각 점의 2D 투영 좌표 계산
    min_dist = float('inf')
    closest_idx = -1
    for i in range(len(embeddings_3d)):
        x2, y2, _ = proj3d.proj_transform(
            embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2],
            ax.get_proj()
        )
        dist = (x2 - click_x) ** 2 + (y2 - click_y) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    if closest_idx >= 0:
        title = df['title'].iloc[closest_idx]
        journal = df['journal'].iloc[closest_idx] if pd.notna(df['journal'].iloc[closest_idx]) else "N/A"
        cluster = df['cluster'].iloc[closest_idx]
        text = f"📄 {title}\n📚 {journal}\n🏷️ Cluster {cluster}"

        annot.set_text(text)
        annot.set_visible(True)
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("button_press_event", on_click)
plt.tight_layout()
plt.show()