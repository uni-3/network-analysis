import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import Word2Vec as word2vec
import pandas as pd
from scipy import sparse

# from sklearn.cluster import SpectralClustering
import sklearn


import random
from itertools import repeat

def get_subnodes(g):
    """
    接続したノードをリストとして抽出
    """
    return [g.subgraph(c) for c in nx.connected_components(g)]


def gen_bipartite_network_projection(df, n1_name="user_id", n2_name="item_id"):
    """
        2部グラフをつくった挙句、その射影グラフを返す
    """
    g = nx.Graph()

    # node、edgeの追加
    g.add_nodes_from(df[n1_name], bipartite=0)
    g.add_nodes_from(df[n2_name], bipartite=1)
    g.add_edges_from([(row[n1_name], row[n2_name]) for idx, row in df.iterrows()])

    top_nodes = {n for n, d in g.nodes(data=True) if d.get("bipartite") == 0}
    bottom_nodes = {n for n, d in g.nodes(data=True) if d.get("bipartite") == 1}

    # 射像グラフ
    g1 = nx.algorithms.bipartite.weighted_projected_graph(g, top_nodes)
    g2 = nx.algorithms.bipartite.weighted_projected_graph(g, bottom_nodes)

    return g1, g2


def get_girvan_newman_com(g):
    """
        girvan_newman法を適用し、ノードの値とコミュニティ番号のdfを返す
    """
    df = pd.DataFrame({"id": g.nodes(), "val": [0] * len(g.nodes())})

    communities_generator = nx.algorithms.community.centrality.girvan_newman(g)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    # print(len(next_level_communities))
    # コミュニティで色分け
    for m in range(len(next_level_communities)):
        for n in next_level_communities[m]:
            df.loc[df["id"] == n, "val"] = m

    print("num of community:", max(df["val"]) + 1)

    return df


from sklearn.cluster import KMeans
import networkx as nx
import numpy as np


class SpectralClustering:
    def __init__(self, k=5):
        self.k = k

        self.km = None
        self.v = None

    def calc_rw_norm_laplacian_matrix(self, X):
        # laplacian
        D = np.diag(np.sum(X, axis=0))
        L = D - X
        self.D = D

        Lrw = np.dot(np.linalg.pinv(D), L)
        self.L = Lrw
        return Lrw

    def get_eigen_topk(self, L):
        """

        Parameters
        ----------
        L

        Returns
        -------
        ee, U
            固有値と固有ベクトルのtopk

        """
        # e, v =  np.linalg.eig(L.A)
        e, v = np.linalg.eig(L)
        self.v = v.real

        # 虚数が出てくる時がある...
        # e_index = np.argsort(-e.real)
        # 昇順に並べた時のindex
        e_index = np.argsort(e)
        # print('topk eigen val', e[e_index])
        ee = e[e_index][: self.k]
        # U = np.array(v[e_index][:self.k].real
        U = np.array(v[e_index][: self.k])

        return ee, U

    def fit(self, G):
        # random walkなラプラシアン用
        # adj_mat = np.array(nx.to_numpy_matrix(G))
        # L = self.calc_rw_norm_laplacian_matrix(adj_mat)

        # 正規化なラプラシアン用
        L = nx.normalized_laplacian_matrix(G).todense()

        _, U = self.get_eigen_topk(L)
        U = U.real

        km = KMeans(n_clusters=self.k, random_state=14)  # init='k-means++',

        km.fit(U)
        self.km = km
        return km

    def predict(self, data=None):
        """固有値ベクトル"""
        v = self.v
        if data:
            v = data
        predictions = self.km.predict(v)

        return predictions


def sk_spectral_clustering(G, n=2):
    # 隣接行列
    mat = nx.to_numpy_matrix(G)
    sparse_mat = sparse.csr_matrix(mat)

    # Cluster
    sc = sklearn.cluster.SpectralClustering(
        n, affinity="precomputed", n_init=100, assign_labels="kmeans", random_state=21
    )
    sc.fit(sparse_mat)

    return sc


# 可視化
def draw_network(
    ng,
    df,
    node_color=None,
    draw_label=True,
    draw_edge=True,
    node_size_apply=None,
    iterations=100,
    edge_thresh=0,
    apply_thresh=None
):
    # removeの副作用回避のため
    g = ng.copy()
    if apply_thresh is not None:
        remove = [node for node, attr in dict(g.nodes(data=True)).items() if attr['apply_count'] < apply_thresh]
        g.remove_nodes_from(remove)
        
    #labels = get_labels(g, df)
    pos = nx.spring_layout(g, k=0.4, iterations=iterations)
    plt.figure(figsize=(14, 14))

    # 閾値を入れてみる
    #edge_width = [d["weight"] * 0.8 for (u, v, d) in g.edges(data=True)]
    edge_width = [d["weight"] * 0.1 if d["weight"] >= edge_thresh else 0 for (u, v, d) in g.edges(data=True)]
    
    if node_size_apply is None:
        node_size_apply = 300
    else:
        node_size_apply = [attr['apply_count']*0.5 for _, attr in g.nodes(data=True)]

    if draw_edge:
        nx.draw_networkx_edges(g, pos,
                               width=edge_width,
                               edge_color='grey',
                               alpha=0.8
                              )
    nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_color,
        node_size=node_size_apply,
        # , cmap=plt.cm.rainbow
        
        #cmap=plt.cm.Pastel1,
    )
    # , cmap=plt.cm.Oranges)
    if draw_label:
        nx.draw_networkx_labels(g, pos=pos, labels=labels, font_family="IPAexGothic")
    plt.axis("off")
    plt.show()


# 渡したnodeを中心としたネットワーク
def draw_ego_network(g, df, node_color=None, node_name=None,
                     node_size_apply=None, edge_thresh=0, apply_thresh=None):
    # node_and_degree = G.degree()
    # (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
    # Create ego graph of main hub
    eg = nx.ego_graph(g, node_name)
    if apply_thresh is not None:
        remove = [node for node, attr in dict(eg.nodes(data=True)).items() if attr['apply_count'] < apply_thresh]
        eg.remove_nodes_from(remove)
        
    #labels = get_labels(eg, df)
    # Draw graph
    pos = nx.spring_layout(eg, k=0.3)
    plt.figure(figsize=(22, 22))
    
    edge_width = [d["weight"] * 0.1 if d["weight"] >= edge_thresh else 0 for (u, v, d) in eg.edges(data=True)]

    if node_size_apply is None:
        node_size_apply = 300
        node_size_ego = 300
    else:
        node_size_apply = [attr['apply_count']*3 for _, attr in eg.nodes(data=True)]
        node_size_ego = [eg.nodes()[node_name]['apply_count']*3]
    
    
    nx.draw(
        eg,
        pos,
        #node_color=list(nx.pagerank(eg).values()),
        node_size=node_size_apply,
        width=edge_width,
        edge_color='grey',
        #alpha=0.5,
        labels=labels,
        font_family="IPAexGothic",
    )
    # Draw ego as large and red
    nx.draw_networkx_nodes(eg, pos,
                           nodelist=[node_name],
                           node_size=node_size_ego,
                           node_color="r")
    plt.show()
    
    return eg


def plot_degree_dist(g):
    degrees = [g.degree(n) for n in g.nodes()]
    plt.hist(degrees)
    plt.show()
    
    
def plot_power(g):
    """
    logを取ってプロットする、近似直線の傾きと切片も算出する
    """
    # log(0)=0とする
    x_log = np.ma.log(range(0, len(nx.degree_histogram(g)))).filled(0)
    y_log = np.ma.log(nx.degree_histogram(g)).filled(0)
    lm_coef = np.polyfit(x_log, y_log, 1)
    # 対数の値と直線近似のプロット
    plt.title(f"傾き: {lm_coef[0]:.3f}, 切片: {lm_coef[1]:.3f}")
    plt.plot(x_log, y_log)
    plt.plot(x_log, np.poly1d(lm_coef)(x_log))
    plt.show()

    
def network_stats(g):
    print(f"""求人2部グラフの射影
    ノード数: {g.number_of_nodes()}
    リンク数: {g.number_of_edges()}
    1node当たりの平均リンク数: {g.number_of_edges()/g.number_of_nodes():.3f}
    クラスタリング係数: {nx.average_clustering(g):.3f}
    次数相関: {nx.algorithms.assortativity.degree_assortativity_coefficient(g):.3f}
    平均パス長: {nx.average_shortest_path_length(g):.3f}
    Qmax: 
    モジュール数
    user transitivity: {nx.transitivity(g):.3f}
    item density: {nx.density(g):.3f}
    """)


# 類似度
def print_sim_node(g, x=3003425278, y=3003475283):
    # x = 3003425278 # チロルチョコの企画事務・デザイン／商品企画・タイアップ企画等
    # y = 3003475283 # 新幹線早期地震防災 株式会社ジェイアール総研エンジニアリング
    print("vertex pair:", x, "and", y)
    print("n of neighbors", x, ":", len(list(g.neighbors(x))))
    print("n of neighbors", y, ":", len(list(g.neighbors(y))))
    print("degree of", x, ":", g.degree(x))
    print("degree of", y, ":", g.degree(y))

    print("common neighbosr:", len(list(nx.common_neighbors(g, x, y))))
    print("Jaccard coefficient:", list(nx.jaccard_coefficient(g, [(x, y)]))[0][2])
    print("Adamic/Adar:", list(nx.adamic_adar_index(g, [(x, y)]))[0][2])
    print(
        "preferential attachment:", list(nx.preferential_attachment(g, [(x, y)]))[0][2]
    )


def print_sim_nodes(g, k=10):
    CN = []  # common neighbors
    JC = []  # jaccard coefficient
    AA = []  # adamic_adar_index
    PA = []  # preferential attachment

    # nodeと次のノード取得
    nodes = list(g.nodes())
    l = g.number_of_nodes()
    for i, x in enumerate(nodes):
        if i < (l - 1):
            y = nodes[i + 1]

        CN.append(tuple([x, y, len(list(nx.common_neighbors(g, x, y)))]))
        JC.append(list(nx.jaccard_coefficient(g, [(x, y)]))[0])
        AA.append(list(nx.adamic_adar_index(g, [(x, y)]))[0])
        PA.append(list(nx.preferential_attachment(g, [(x, y)]))[0])

    # top k
    print("vertex pair:", x, "and", y)
    print("common neighbors")
    print(sorted(CN, key=lambda x: x[2], reverse=True)[:k])
    print("Jaccard coefficient")
    print(sorted(JC, key=lambda x: x[2], reverse=True)[:k])
    print("Adamic/Adar")
    print(sorted(AA, key=lambda x: x[2], reverse=True)[:k])
    print("preferential attachment")
    print(sorted(PA, key=lambda x: x[2], reverse=True)[:k])
