import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import Ridge

def powernorm(pfs, alpha=0.5):
    pfs = np.sign(pfs) * (np.abs(pfs)**alpha)    
    return normalize(pfs, axis=1)

class Encoding:
    """Provides a base class for aggregating local descriptors"""
    
    def __init__(self, norms=['powernorm'], pn_alpha=0.5, num_samples=-1):
        if type(norms) != list:
            norms = [norms]
        self.norms = norms
        self.pn_alpha = pn_alpha
        self.num_samples = num_samples

    def sample(self, pf):
        if self.num_samples == -1:
            return pf

        s = pf.shape[0]
        if s < self.num_samples:
            return pf
        idx = np.linspace(0, s-1, self.num_samples).astype('int')
        return pf[idx]

    def normalize(self, pfs):
        if 'powernorm' in self.norms:
            pfs = powernorm(pfs, alpha=self.pn_alpha)
        if 'l2' in self.norms:
            pfs = normalize(pfs, axis=1)
        if 'intranorm' in self.norms:
            pfs = normalize(normalize(pfs, axis=0), axis=1)
        return pfs


class SumPooling(Encoding):

    def encode(self, pfs):
        # pfs expected to be a list
        pfs = np.array([np.sum(self.sample(pf), axis=0) for pf in pfs])
        pfs = self.normalize(pfs)
        return pfs

class MaxPooling(Encoding):
    def encode(self, pfs):
        pfs = np.array([np.max(self.sample(pf),axis=0) for pf in pfs])
        pfs = self.normalize(pfs)
        return pfs

class LSEPooling(Encoding):
    def encode(self, pfs):
        pfs = np.array([np.log(np.sum(np.exp(self.sample(pf)),axis=0)) for pf in pfs])
        pfs = self.normalize(pfs)
        return pfs

class GMP(Encoding):

    def __init__(self, norms=['powernorm'], pn_alpha=0.5, alpha=1000):
        super().__init__(norms=norms, pn_alpha=pn_alpha)
        self.alpha = alpha

    def encode(self, pfs):
        # pfs expected to be a list

        def aggregate(phi):
            scaler = StandardScaler()
            phi = scaler.fit_transform(phi)
            clf = Ridge(fit_intercept=False, alpha=self.alpha)
            N, KD = phi.shape
            clf.fit(phi, np.ones(N))

            v = clf.coef_.reshape(1,-1)
            return v

        pfs = [aggregate(self.sample(pf)) for pf in pfs]
        pfs = self.normalize(pfs)
        return pfs

if __name__ == '__main__':
    import numpy as np
    feats = np.random.rand(1000,6400)
    pf = SumPooling().encode([feats])
    print(pf)
    print(pf[0].shape)
    print(np.linalg.norm(pf[0]))





    