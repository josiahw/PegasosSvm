
import numpy, time, numpy.typing, random
from collections import defaultdict



class SvmSolver:
    dtype = numpy.float32

    verbose = False
    
    def __init__(self, C, kernel, dtype, data_size, dimensionality):
        self.C = C
        self.kernel = kernel
        self.dimensionality = dimensionality
        self.data_size = data_size
        self.dtype = dtype

    def Solve(self, X, y):
        abstract

class PegasosSolver(SvmSolver):

    resize_step = 100
    support_vectors = None
    support_weights = None
    support_targets = None

    
    def __init__(self, C, dtype, kernel, data_size, dimensionality):
        """
        Init the solver with a cost variable, internal data type, kernel, training set size, and number of dimensions
        """
        super(PegasosSolver, self).__init__(C, kernel, dtype, data_size, dimensionality)
        self.gamma = 1. / data_size / self.C
        self.T = int(1. / self.gamma / self.C)

        # TODO: should these be inited below?
        self.sv_map = defaultdict(lambda: None)
        self.alpha_indices = []
        self.current_size = 0

    @staticmethod
    def _Error(kernel_result: numpy.typing.ArrayLike, support_weights: numpy.typing.ArrayLike, gamma: float, target: numpy.typing.ArrayLike):
        return target / gamma * (numpy.sum(support_weights * kernel_result, axis=-1))

    def Solve(self, X: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike, stride: int = 10):
        t0 = time.time()
        t = 0
        while t < self.T:
            start_index = t % len(X)
            end_index = min(len(X), start_index + stride)
            self.SolveIncremental(t, numpy.arange(start_index,end_index), X[start_index:end_index], y[start_index:end_index])
            t += end_index - start_index
        if self.verbose:
            print(f"Solve {self.current_size} SVs in {time.time()-t0:.2f}s")
            t0 = time.time()
        self.t = t
        return self.support_vectors[:self.current_size].copy(), y[self.alpha_indices] * self.support_weights[:self.current_size], self.alpha_indices

    def SolveIncremental(self, t: int, selected_values: numpy.typing.ArrayLike, X: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike):
        """
        Pegasos algorithm - inner loop iteration
        """
        X = X.reshape((-1,self.dimensionality))
        if self.support_vectors is None:
            self.sv_map[selected_values[0]] = 0
            self.alpha_indices.append(selected_values[0])
            self.support_vectors = X[0].reshape((1,self.dimensionality)).copy()
            self.support_weights = numpy.array([1], dtype=self.dtype)
            self.support_targets = numpy.array([y[0]], dtype=self.dtype)
            self.current_size = 1

        k = self.support_targets[:self.current_size] * self.kernel(X, self.support_vectors[:self.current_size])
        lambda_final = self.gamma * (t + len(y)/2)
        error = self._Error(
                    k, 
                    self.support_weights[:self.current_size], 
                    lambda_final, 
                    y
                )
        if (error < 1).any():
            for i, selected_value in enumerate(selected_values):
                if (error[i] < 1):
                    self.sv_index = self.sv_map[selected_value] or self.current_size
                    if self.sv_index == len(self.support_weights):
                        # CASE: need to resize
                        self.support_weights.resize(self.support_weights.shape[0] + self.resize_step)
                        self.support_targets.resize(self.support_targets.shape[0] + self.resize_step)
                        self.support_vectors.resize((self.support_vectors.shape[0] + self.resize_step, self.support_vectors.shape[1]))
                    if self.sv_index == self.current_size:
                        # CASE: add new stuff
                        self.sv_map[selected_value] = self.current_size
                        self.alpha_indices.append(selected_value)
                        self.support_vectors[self.sv_index] = X[i]
                        self.support_targets[self.sv_index] = y[i]
                        self.current_size += 1
                    self.support_weights[self.sv_index] += 1