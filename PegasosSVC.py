# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 20:40:56 2016

@author: josiahw
"""
import numpy, time, numpy.linalg, numpy.typing
from sklearn.metrics.pairwise import chi2_kernel, polynomial_kernel
from MultiThreadedExecutor import MultiThreadedExecutor
from PegasosSolver import PegasosSolver
import numpy
from scipy import sparse

class jaccard_kernel:

    def __init__(self):
        pass

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
        # TODO: handle zero padding
        return (numpy.sum(X.reshape((1, X.shape[0], X.shape[1])) == Y.reshape((Y.shape[0], 1, Y.shape[1])), axis=-1) / X.shape[1]).squeeze().reshape((Y.shape[0],X.shape[0])).T
    
    def InnerKernel(self, X, Y):
        """
        Assumes X or Y is singular, or they are of the same length.
        Returns a 1d array of paired distances
        """
        Z = X == Y
        return Z / X.shape[1] 

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
                 kernel = rbf_kernel(0.0001),
                 dataset_size = None,
                 dimensionality = None
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
        self.dimensionality = dimensionality
        self.dataset_size = dataset_size
        if not dataset_size is None:
            self.solver = PegasosSolver(self.C, self.dtype, self.kernel, dataset_size, dimensionality)

    def update(self, indices, X, y):
        """
        This assumes enough info has been given at the outset
        """
        self.solver.SolveIncremental(indices, X, y)

    def fit(self, X: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike):
        """
        Fit to data X with labels y.
        """
        t0 = time.time()
        self.solver = PegasosSolver(self.C, X.dtype, self.kernel, X.shape[0], X.shape[1])
        sv, alphas, sv_indices = self.solver.Solve(X, y)
        if self.verbose:
            print(f"Solved {len(alphas)} SVs in {time.time()-t0:.2f}")
        self.a = alphas
        self.sv = sv
        self.indices = sv_indices

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
                kernel = rbf_kernel(0.00001),
                classes = None,
                dataset_size = None,
                dimensionality = None): # TODO: fix kernel default
        self.C = C
        self.kernel = kernel
        self.dtype = dtype
        self.classes = classes
        self.dimensionality = dimensionality
        self.dataset_size = dataset_size
        self.predictors = None
        if not classes is None and not dataset_size is None:
            self.predictors = [SvcPredictor(self.C, self.dtype, self.kernel, self.dataset_size, self.dimensionality) for c in classes]


    def update(self, indices, X, y, threadDispatcher = None):
        """
        This assumes enough info has been given at the outset
        """
        data_func = lambda x: x[0].update(indices, X, x[1])
        data = [(self.predictors[i], (y == c)*2. - 1.) for i, c in enumerate(self.classes)]
        if threadDispatcher is None:
            for d in data:
                data_func(d)
        else:
            threadDispatcher.exec(data_func, data)


    def fit(self, X, y):
        if self.classes is None:
            self.classes = numpy.unique(y)
        sv_indices = set()
        t0 = time.time()
        data_func = lambda x: x[0].fit(X, x[1])
        if self.predictors is None:
            self.predictors = [SvcPredictor(self.C, self.dtype, self.kernel, X.shape[0], X.shape[1]) for c in self.classes]
        data = [(self.predictors[i], (y == c)*2. - 1.) for i, c in enumerate(self.classes)]
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
    