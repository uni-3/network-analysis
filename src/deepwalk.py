from gensim.models import Word2Vec as word2vec

import random

# word2vec model
min_count=0
size=2
window=5
hs=1
workers=1

def random_walk(G, start_node, len_walk):
    random.seed(0)
    path = [str(start_node)]
    current_node = start_node

    for _ in range(len_walk):
        neighbors = list(G.neighbors(current_node))

        if (len(neighbors) == 0):
            break

        current_node = random.choice(neighbors)
        path.append(str(current_node))

    return path


def build_walks(G, walk_per_node=20, path_len=10):
    """
        max_paths: あるノードで試行する回数
        path_len: 歩く回数
    """

    print(f"Building walks with: walk per node = {walk_per_node}, path_length = {path_len}")
    walks = []
    nodes = list(G)
    rand = random.Random(0)
    for _ in range(walk_per_node):
        # 毎回、ノードの順序をランダムにする

        rand.shuffle(nodes)
        # map版
        # walks = walks + list(map(random_walk, repeat(G), nodes, repeat(path_len)))
        # list版
        #walks = walks + [random_walk(G, n, path_len) for n, v in G.nodes(data=True)]
        walks = walks.append([random_walk(G, n, path_len) for n, v in G.nodes(data=True)])

    print("Completed")

    return walks


def build_walk_model(walks):
    model = word2vec(walks, min_count=min_count
                     , size=size, window=window
                     , hs=hs, workers=workers)

    return model