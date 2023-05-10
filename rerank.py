import argparse, os

import torch
import numpy as np

from reranking import gnn, krnn, sgr

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_path', type=str, help='path to .npy file storing the embeddings which should be reranked')
parser.add_argument('--algorithm', type=str, default="sgr", help='reranking algorithm. options: sgr, gnn, krnn')

parser.add_argument('--k', type=int, default=2, help='krnn: k reciprocal nearest neighbors used for reranking, sgr: number of neighbors used for SGR, see paper')
parser.add_argument('--k1', type=int, default=1, help='gnn only: k1')
parser.add_argument('--k2', type=int, default=2, help='gnn only: k2')
parser.add_argument('--layers', type=int, default=1, help="gnn/sgr only: number of layers for aggregation")
parser.add_argument('--gamma', type=float, default=0.4, help="sgr only: gamma weight, see paper")

parser.add_argument('--save_path', type=str, default='results', help='directory where reranked embeddings should be save')
parser.add_argument('--run_name', type=str, default='reranking', help='identifier for reranked embeddings')

args = parser.parse_args()

# X ... embeddings
X = torch.tensor(np.load(args.embedding_path))

if args.algorithm == 'sgr':
    X_reranked, _ = sgr.sgr_reranking(X, args.k, layer=args.layers, gamma=args.gamma)
elif args.algorithm == 'krnn':
    X_reranked = krnn.kRNN(X, args.k)
elif args.algorithm == 'gnn':
    X_reranked, _ = gnn.gnn_reranking(X, args.k1, args.k2, layer=args.layers)
else:
    raise("Unknown reranking!")

np.save(os.path.join(args.save_path, f"{args.run_name}_{args.algorithm}.npy"), X.numpy())
