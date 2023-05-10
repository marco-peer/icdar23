import torch

def kRNN(X, k):
    S = torch.mm(X, X.t())
    # initial ranking list
    _, initial_rank = S.topk(k=S.shape[0], dim=-1, largest=True, sorted=True)
    kNN = initial_rank[:, 1:k+1]

    reranked = torch.zeros(X.shape)
    for i in range(X.shape[0]):
        feat = X[i]
        nn = kNN[i]


        rnn = [X[j] for j in nn if i in kNN[j]]

        if rnn:
            rnn = torch.concat(rnn).view(-1, rnn[0].shape[0])
            reranked[i] = (feat + torch.sum(rnn, dim=0)) / (rnn.shape[0] + 1)
        else:
            reranked[i] = feat

    reranked_norm = torch.norm(reranked, p=2, dim=1, keepdim=True)
    reranked = reranked.div(reranked_norm.expand_as(reranked))  
    return reranked