# coding: utf-8

import numpy as np
import networkx as nx
import scipy.io
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from multiprocessing import cpu_count
import concurrent.futures
import os
from collections import defaultdict
from time import perf_counter
from datetime import timedelta
import argparse



def timer(msg: str):
    def inner(func):
        def wrapper(*args, **kwargs):
            t1 = perf_counter()
            ret = func(*args, **kwargs)
            t2 = perf_counter()
            print("Time elapsed for " + msg + " ----> " + str(timedelta(seconds=t2 - t1)))
            print("\n---------------------------------------\n")
            return ret

        return wrapper

    return inner


class Graph():

    def __init__(self, graph: nx.Graph, probs: list, p: float, q: float, walks_par_node: int, walk_len: int):

        self.graph = graph
        self.probs = probs
        self.p = p
        self.q = q
        self.walks_par_node = walks_par_node
        self.walk_len = walk_len

    def compute_probabilities_conc(self, source_node):

        G = self.graph
        #for source_node in G.nodes():

        for current_node in list(G[source_node]):

            probs_ = list()
            for neighbor in list(G[current_node]):
                # 同じやつ
                if source_node == neighbor:
                    prob_ = G[current_node][neighbor].get('weight', 1) * (1 / self.p)
                # 1つ離れている
                elif neighbor in G.neighbors(source_node):
                    prob_ = G[current_node][neighbor].get('weight', 1)
                # 2つ以上離れている
                else:
                    prob_ = G[current_node][neighbor].get('weight', 1) * (1 / self.q)

                probs_.append(prob_)

            self.probs[source_node]['prob'][current_node] = probs_ / np.sum(probs_)

        return


    @timer('Computing probabilities')
    def compute_probabilities(self):

        G = self.graph
        for source_node in G.nodes():

            for current_node in list(G[source_node]):

                probs_ = list()
                for neighbor in list(G[current_node]):
                    # 同じやつ
                    if source_node == neighbor:
                        prob_ = G[current_node][neighbor].get('weight', 1) * (1 / self.p)
                    # 1つ離れている
                    elif neighbor in G.neighbors(source_node):
                        prob_ = G[current_node][neighbor].get('weight', 1)
                    # 2つ以上離れている
                    else:
                        prob_ = G[current_node][neighbor].get('weight', 1) * (1 / self.q)

                    probs_.append(prob_)

                self.probs[source_node]['prob'][current_node] = probs_ / np.sum(probs_)

        return


    def generate_walk(self, node):

        G = self.graph
        walk = [node]
        walk_neighbors = list(G[node])
        if len(walk_neighbors) == 0:
            return walk

        first_step = np.random.choice(walk_neighbors)
        walk.append(first_step)

        for _ in range(self.walk_len - 2):
            walk_neighbors = list(G[walk[-1]])
            if len(walk_neighbors) == 0:
                break

            probabilities = self.probs[walk[-2]]['prob'][walk[-1]]
            next_step = np.random.choice(walk_neighbors, p=probabilities)
            walk.append(next_step)

        np.random.shuffle(walk)

        return walk

    def generate_node_walks_conc(self, node):
        G = self.graph
        walks = list()
        #for node in G.nodes():
        for i in range(self.walks_par_node):
            walk = self.generate_walk(node)
            walks.append(walk)

        np.random.shuffle(walks)
        #walks = [list(map(str, walk)) for walk in walks]
        walks = [list(map(str, walk)) for walk in walks]

        return walks

    @timer('Node Walks')
    def generate_node_walks(self):
        G = self.graph
        walks = list()
        for node in G.nodes():
            for i in range(self.walks_par_node):
                walk = self.generate_walk(node)
                walks.append(walk)

        np.random.shuffle(walks)
        #walks = [list(map(str, walk)) for walk in walks]
        walks = [list(map(str, walk)) for walk in walks]

        return walks



def read_graph(input_path, directed=False):
    if (input_path.split('.')[-1] == 'edgelist'):
        G = nx.read_edgelist(input_path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())

    elif (input_path.split('.')[-1] == 'mat'):
        edges = list()
        mat = scipy.io.loadmat(input_path)
        nodes = mat['network'].tolil()
        G = nx.DiGraph()
        for start_node, end_nodes in enumerate(nodes.rows, start=0):
            for end_node in end_nodes:
                edges.append((start_node, end_node))

        G.add_edges_from(edges)

    else:
        import sys
        sys.exit('Unsupported input type')

    if not directed:
        G = G.to_undirected()

    probs = defaultdict(dict)
    for node in G.nodes():
        probs[node]['prob'] = dict()

    print(nx.info(G) + "\n---------------------------------------\n")
    return G, probs

def init_probs(G, directed=False):
    # ノードの遷移確率を入れておくやつ
    probs = defaultdict(dict)
    for node in G.nodes():
        probs[node]['prob'] = dict()

    return probs


@timer('Generating embeddings')
def generate_embeddings(corpus, dimensions, window, workers, output_file, p=0.5, q=0.5):
    model = Word2Vec(corpus, size=dimensions, window=window, min_count=0, sg=1, workers=workers)
    # model.wv.most_similar('1')
    #w2v_emb = model.wv

    if output_file == None:
        import re
        output_file = 'embeddings/' + 'n2v' + '_embeddings_' + 'dim-' + str(dimensions) + '_p-' + str(p) + '_q-' + str(q) + '.model'

    print('Saved model to : ', output_file)
    #w2v_emb.save_word2vec_format(output_file)
    model.save(output_file)
    #model.wv.save_word2vec_format(output_file)

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="node2vec options")

    parser.add_argument('--output', nargs='?', default='n2v.model'
                        , help='Path for saving output gensim word2vec model file')

    parser.add_argument('--p', default='1.0', type=float, help='Return parameter')

    parser.add_argument('--q', default='1.0', type=float, help='In-out parameter')

    parser.add_argument('--walks', default=10, type=int, help='Walks per node')

    parser.add_argument('--length', default=80, type=int, help='Length of each walk')

    parser.add_argument('--d', default=128, type=int, help='Dimension of output embeddings')

    parser.add_argument('--window', default=10, type=int, help='Window size for word2vec')

    parser.add_argument('--workers', default=1, type=int,
                        help='Number of workers to assign for random walk and word2vec')

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Specify if graph is directed. Default is undirected')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def main(args):

    Graph, init_probabilities = read_graph(args.input, args.directed)
    #G =
    #propb = init_probs(G)

    G = Graph(Graph, init_probabilities, args.p, args.q, args.walks, args.length, args.workers)
    #G.compute_probabilities()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for source_node in G.graph.nodes():
            executor.submit(G.compute_probabilities_conc, source_node)
    #walks = G.generate_random_walks()
    walks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for node in G.graph.nodes():
            walks_map = executor.submit(G.generate_random_walks_conc, node)

            walks.append(walks_map)

    walks = np.array(walks).flatten()
    model = generate_embeddings(walks, args.d, args.window, args.workers
                                , args.p, args.q, args.input, args.output)
    embeddings = model.wv


if __name__ == '__main__':
    args = parse_args()
    main(args)
