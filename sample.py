import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 1. 데이터베이스 로드
conn = sqlite3.connect('papers.db')
query = """
SELECT id, title, abstract
FROM papers
"""

df = pd.read_sql_query(query, conn)

# 3. CSV로 저장
df.to_csv("papers_abstract.csv", index=False, encoding="utf-8-sig")

# 4. 출력해서 테이블처럼 보기
print(df.head())

conn.close()
quit()
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

# 4. t-SNE 2D 압축 (sample 포함하여 함께 변환)
tsne = TSNE(n_components=2, random_state=42)
all_2d = tsne.fit_transform(all_embeddings)

embeddings_2d = all_2d[:-1]      # DB 논문들의 2D 좌표
sample_2d = all_2d[-1]           # sample의 2D 좌표

# 5. 시각화 - 클러스터별 색상 + sample 빨간점 + 호버
fig, ax = plt.subplots(figsize=(14, 9))

# 클러스터별 색상 팔레트
cmap = plt.colormaps['tab10']

# DB 논문 scatter (클러스터 색상)
scatter = ax.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1],
    c=df['cluster'], cmap='tab10', alpha=0.6, s=40,
    edgecolors='white', linewidths=0.3,
)

# sample_title 빨간 점 (크게, 별 모양)
ax.scatter(
    sample_2d[0], sample_2d[1],
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

ax.set_title("Paper Semantic Map (SPECTER 2 + t-SNE, K=5 Clusters)")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")

# 호버용 annotation 생성
annot = ax.annotate(
    "", xy=(0, 0), xytext=(15, 15),
    textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", alpha=0.95),
    fontsize=9,
    wrap=True,
)
annot.set_visible(False)


def on_hover(event):
    """마우스 호버 시 해당 논문의 제목, 저널, 클러스터 정보를 표시"""
    if event.inaxes != ax:
        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()
        return

    cont, ind = scatter.contains(event)
    if cont:
        idx = ind["ind"][0]
        pos = scatter.get_offsets()[idx]
        annot.xy = pos

        title = df['title'].iloc[idx]
        journal = df['journal'].iloc[idx] if pd.notna(df['journal'].iloc[idx]) else "N/A"
        cluster = df['cluster'].iloc[idx]
        text = f"📄 {title}\n📚 {journal}\n🏷️ Cluster {cluster}"

        annot.set_text(text)
        annot.set_visible(True)
        fig.canvas.draw_idle()
    else:
        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", on_hover)
plt.tight_layout()
plt.show()