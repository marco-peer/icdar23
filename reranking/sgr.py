import torch

def build_adjacency_matrix(init_rank, S, gamma=0.5):
    A = torch.eye(init_rank.shape[0])
    for i in range(A.shape[0]):
        A[i, init_rank[i, :]] =  torch.exp(-torch.pow(1-S[i, init_rank[i,:]], 2) / gamma)
    return A

def gnn_propagate(A, S, r_k2):
    A_ = torch.zeros(A.shape)
    for i in range(A.shape[0]):
        hi =  A[i, :]
        k2nn = r_k2[i, 1:]
        hi += torch.sum(torch.mul(S[i, k2nn], A[:, k2nn]), dim=-1)
        A_[i, :] = hi

    return A_

def sgr_reranking(X, k, layer=2, gamma=0.5):

    S = torch.mm(X, X.t())
    del X

    # initial ranking list
    _, initial_rank = S.topk(k=S.shape[0], dim=-1, largest=True, sorted=True)

    # stage 1
    A = build_adjacency_matrix(initial_rank, S, gamma=gamma)   

    # stage 2
    for i in range(layer):
        A = (A + A.T) / 2

        _, r_k2 = A.topk(k=(k+1), dim=-1, largest=True, sorted=True)

        A = gnn_propagate(A, S, r_k2)
        A_norm = torch.norm(A, p=2, dim=1, keepdim=True)
        A = A.div(A_norm.expand_as(A))  

    reranked = A
    cosine_distance = 1 - torch.mm(A, A.t())
    return reranked.cpu(), cosine_distance.cpu()
