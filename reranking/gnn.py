import torch

def pow2(vec):
    return torch.pow(vec, 2)

def build_adjacency_matrix(init_rank):
    A = torch.eye(init_rank.shape[0])
    for i in range(A.shape[0]):
        A[i, init_rank[i]] = 1
    return A

def gnn_propagate(A, S, k2nn, func):
    A_ = torch.zeros(A.shape)
    for i in range(A.shape[0]):
        hi =  A[i, :]
        hi += torch.sum(torch.mul(func(S[i, k2nn[i]]), A[:, k2nn[i]]), dim=-1)
        A_[i, :] = hi

    return A_

def gnn_reranking(X, k1, k2, layer=2, func=pow2):

    S = torch.mm(X, X.t())
    del X

    # initial ranking list
    _, initial_rank = S.topk(k=k1+1, dim=-1, largest=True, sorted=True)

    # stage 1
    A = build_adjacency_matrix(initial_rank)   

    # stage 2
    for i in range(layer):
        A = (A + A.T) / 2

        _, k2nn = A.topk(k=(k2+1), dim=-1, largest=True, sorted=True)

        A = gnn_propagate(A, S, k2nn, func=func)
        A_norm = torch.norm(A, p=2, dim=1, keepdim=True)
        A = A.div(A_norm.expand_as(A))  

    reranked = A
    cosine_distance = 1 - torch.mm(A, A.t())
    return reranked.cpu(), cosine_distance.cpu()