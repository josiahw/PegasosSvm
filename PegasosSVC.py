# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 20:40:56 2016

@author: josiahw
"""
import numpy, time, numpy.linalg, numpy.typing
from sklearn.metrics.pairwise import chi2_kernel, polynomial_kernel
from MultiThreadedExecutor import MultiThreadedExecutor
from PegasosSolver import PegasosSolver
import torch
from scipy import sparse

class rbf_kernel:

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, Y):
        """
        Assumes X and Y are of shape (instances, dimensionality)
        """
        if X.shape[0] == 1 or Y.shape[0] == 1:
            return self.InnerKernel(X, Y)
        return self.OuterKernel(X, Y)
    
    def OuterKernel(self, X, Y):
        """
        Returns X * Y matrix of all combinations of distances
        """
        XX = numpy.sum(X**2,axis=-1)
        YY = numpy.sum(Y**2,axis=-1)
        XY = numpy.dot(X,Y.T)
        Z = XX.reshape((-1,1)) + YY.reshape((1,-1)) - 2 * XY
        #Z = scipy.spatial.distance.cdist(X, Y, metric='sqeuclidean')
        return numpy.exp(-self.gamma * Z)
    
    def InnerKernel(self, X, Y):
        """
        Assumes X or Y is singular, or they are of the same length.
        Returns a 1d array of paired distances
        """
        Z = numpy.sum(numpy.square(numpy.subtract(X, Y)), axis=1)
        return numpy.exp(-self.gamma * Z)

@torch.jit.script
def rbf_kernel_torch(X: torch.Tensor, gamma: torch.Tensor, support_vectors: torch.Tensor, alphas: torch.Tensor):
    return 2 * alphas @ torch.exp(-gamma * ((X - support_vectors)**2).sum(1).squeeze()) - 1

class rbf_model(torch.nn.Module):
    """
    A torch impl of the rbf_kernel model - for use in backprop
    """
    # define model elements
    def __init__(self, support_vectors, alphas, gamma):
        super(rbf_model, self).__init__()
        self.gamma = torch.tensor(gamma, requires_grad=False)
        self.support_vectors = torch.nn.Parameter(torch.tensor(support_vectors, requires_grad=True), requires_grad=True)
        self.alphas = torch.tensor(alphas, requires_grad=False)

    # forward propagate input
    def forward(self, X: torch.Tensor):
        return torch.clip(rbf_kernel_torch(X, self.gamma, self.support_vectors, self.alphas),-1,1)

class SvcPredictor:
    """
    A base single class (1/-1) SVC predictor. Not intended to be exposed to users.
    """
    C = None
    sv = None
    kernel = None
    verbose = True

    def __init__(self,
                 C,
                 dtype = numpy.float32,
                 kernel = rbf_kernel(0.0001)
                 ):
        """
        The parameters are:
         - C: SVC cost
         - tolerance: gradient descent solution accuracy
         - kernel: the kernel function do use as k(a, b, *kwargs)
         - kwargs: extra parameters for the kernel
        """
        self.C = C
        self.kernel = kernel
        self.dtype = dtype

    def fit(self, X: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike, back_fit = False):
        """
        Fit to data X with labels y.
        """
        t0 = time.time()
        solver = PegasosSolver(self.C, X.dtype, self.kernel, X.shape[0], X.shape[1])
        sv, alphas, sv_indices = solver.Solve(X, y)
        del solver
        if self.verbose:
            print(f"Solved {len(alphas)} SVs in {time.time()-t0:.2f}")
        self.a = alphas
        self.sv = sv
        self.indices = sv_indices

        if back_fit:
            self.back_fit(X,y)

    def back_fit(self, X: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike):
        """
        TODO: implement predict_proba with torch types and hold all else constant while doing grad descent on support vectors
        """
        model= rbf_model(self.sv, self.a, self.kernel.gamma)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), 0.05)
        batch_size = 1024
        counter = 0
        X = torch.tensor(X, requires_grad=False)
        # XXX: this is RBF kernel only - TODO: define a set of acceptable kernels with their torch funcs
        for i in range(5):
            l = 0
            t0 = time.time()
            for t, x in enumerate(X):
                result = model.forward(x)
                loss = (result - y[t])**2
                l += loss.item()
                loss.backward()
                if counter % batch_size == batch_size-1:
                    optimizer.step()
                counter += 1
            if self.verbose:
                print(f"Epoch in {time.time()-t0:.2f}s - Error: {l/len(X)}")
        
        self.sv = model.support_vectors.detach().numpy()

    def _predict_raw(self, X: numpy.typing.ArrayLike):
        """
        For SVClustering, we need to calculate radius rather than bias.
        """
        # multithreading speeds up 
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        threadDispatcher = MultiThreadedExecutor()
        data_func = lambda x: numpy.subtract(numpy.asarray(2 * self.a.dot(self.kernel(x, self.sv).squeeze().T).T), self.kernel.InnerKernel(x, x).reshape((1,-1))).squeeze()
        Z = threadDispatcher.fill(data_func, X, (len(X),))
        return Z
    
    def predict(self, X: numpy.typing.ArrayLike):
        """
        This assumes 1/-1 label outputs
        """
        probas = self._predict_raw(X)
        print(probas)
        classes = numpy.ones(len(probas), dtype=numpy.int64)
        classes[probas < 0] = -1
        return classes


class PegasosSVC:
    models = []
    sv = None
    a = None
    verbose = True

    def __init__(self,
                C,
                dtype = numpy.float32,
                kernel = rbf_kernel(0.00001)): # TODO: fix kernel default
        self.C = C
        self.kernel = kernel
        self.dtype = dtype

    def fit(self, X, y, back_fit = False):
        self.classes = numpy.unique(y)
        sv_indices = set()
        t0 = time.time()
        data_func = lambda x: x[0].fit(X, x[1], back_fit=back_fit)
        data = [(SvcPredictor(self.C, self.dtype, self.kernel), (y == c)*2. - 1.) for c in self.classes]
        #turns out threading is slower. Probably due to cache
        #threadDispatcher = MultiThreadedExecutor()
        #threadDispatcher.exec(data_func, data)
        for d in data:
            data_func(d)
        sv_total_count = 0
        for model, labels in data:
            sv_indices.update(model.indices)
            sv_total_count += len(model.indices)

        # if not enough svs are shared, it's faster to not fuse models
        if len(sv_indices) > 0.5 * sv_total_count:
            if self.verbose:
                print("Models share too few SVs, will not merge")
            self.models = [d[0] for d in data]
        else:
            # fuse predictors for faster inference:
            sv_indices = list(sv_indices)
            sv_indices.sort()
            self.sv = X[sv_indices].copy()

            if back_fit:
                for out_index,orig_index in enumerate(sv_indices):
                    fitted_svs = []
                    for model, labels in data:
                        if orig_index in model.indices:
                            fitted_svs.append(model.sv[model.indices.index(orig_index)])
                    self.sv[out_index] = sum(fitted_svs)/len(fitted_svs)

            # create a 2D shared weight matrix to calculate all class proba's at once
            self.a = numpy.zeros((len(data),len(sv_indices)), dtype=self.dtype)
            index_map = {k: i for i, k in enumerate(sv_indices)}
            for i in range(len(data)):
                self.a[i,[index_map[j] for j in data[i][0].indices]] = data[i][0].a
            # The shared weight matrix is often quite sparse as not all sv's are shared. It saves memory and compute to formalise that
            self.a = sparse.csr_matrix(self.a)
            del data
        if self.verbose:
            print(f"Solved {len(sv_indices)} SVs in {time.time()-t0:.2f}")
        
        return self

    def _predict_raw(self, X: numpy.typing.ArrayLike):
        # if we didn't combine models, get each prediction individually:
        if len(self.models):
            return numpy.array([m._predict_raw(X) for m in self.models]).T

        # else do combined model kernel
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        threadDispatcher = MultiThreadedExecutor()
        data_func = lambda x: numpy.subtract(numpy.asarray(2 * self.a.dot(self.kernel(x, self.sv).squeeze().T).T), self.kernel.InnerKernel(x, x).reshape((-1,1)))
        return threadDispatcher.fill(data_func, X, (len(X), self.a.shape[0]))

    def predict_proba(self, X: numpy.typing.ArrayLike):
        """
        Standard formula for multiclass probabilities for SVC
        """
        result = numpy.exp(self._predict_raw(X))
        return result / numpy.sum(result, axis=1).reshape((-1,1))

    def predict(self, X):
        """
        Return class predictions for data X
        """
        return self.classes[numpy.argmax(self._predict_raw(X), axis=1)]


if __name__ == '__main__':
    from test_datasets import test_mnist, test_cifar10
    #import matplotlib
    #data,labels = load_moons(10000)
    """
    #parameters can be sensitive, these ones work for two moons
    C = 0.025
    clss = MultiClassSVM(C,numpy.float32,rbf_kernel,gamma=13)
    t0 = time.time()
    clss.fit(data, labels)
    print(f"fit in {time.time()-t0:.2f} seconds")

    #check assigned classes for the two moons as a classification error
    t0 = time.time()
    t = clss.predict(data)
    print(f"predicted in {time.time()-t0:.2f} seconds")
    print ("Error", numpy.sum((labels-t)**2) / float(len(data)))
    
    """
    print("Testing MNIST - SVC")
    error = test_mnist()
    print ("Error", error)
    print("Testing Cifar10 - SVC")
    error = test_cifar10()
    print ("Error", error)
    
    if False: # compare to sklearn svm
        print("testing MNIST - LibSVM")
        from sklearn.datasets import fetch_openml
        from sklearn.utils import check_random_state
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = load_data()
        from sklearn import svm
        clss = svm.SVC(C=C,gamma=gamma)
        t0 = time.time()
        clss.fit(X_train,y_train)
        print(f"sklearn fit in {time.time()-t0:.2f} seconds")
        t0 = time.time()
        t = clss.predict(X_test)
        print(f"predicted in {time.time()-t0:.2f} seconds")
        print ("Error", numpy.sum((y_test!=t)**2) / float(len(data)))
    