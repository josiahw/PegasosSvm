
import numpy, time, numpy.typing, random
from collections import defaultdict



class SvmSolver:
    dtype = numpy.float32

    verbose = False
    
    def __init__(self, C, kernel, dtype, data_size, dims):
        self.C = C
        self.kernel = kernel
        self.dimensionality = dims
        self.data_size = data_size
        self.dtype = dtype

    def Solve(self, X, y):
        abstract

class PegasosSolver(SvmSolver):

    # TODO: make stateful so solver can stream
    def __init__(self, C, dtype, kernel, data_size, dims, use_regression_target = False):
        super(PegasosSolver, self).__init__(C, kernel, dtype, data_size, dims)
        self.sv_map = defaultdict(lambda: None)
        self.current_size = 1
        self.support_vectors = None #X[0].reshape((1,-1)).copy()
        self.support_weights = None #numpy.array([0], dtype=self.dtype)
        self.support_targets = None #numpy.array([y[0]], dtype=self.dtype)
        self.alpha_indices = []
        self.gamma = 1. / data_size / self.C
        self.T = int(1. / self.gamma / self.C)
        self.resize_step = 100
        self.use_regression_target = use_regression_target

    @staticmethod
    def _Error(kernel_result: numpy.typing.ArrayLike, support_weights: numpy.typing.ArrayLike, gamma: float, target = 1, offset = 0):
        return target / gamma * (numpy.sum(support_weights * kernel_result, axis=-1) - offset)

    def Solve(self, X: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike, stride = 1):
        t0 = time.time()
        t = 0
        self.bias = 0
        while t < self.T:
            start_index = t % len(X)
            end_index = min(len(X), start_index + stride)
            self.SolveIncremental(t, list(range(start_index,end_index)), X[start_index:end_index], y[start_index:end_index])
            t += end_index - start_index
        if self.verbose:
            print(f"Solve {self.current_size} SVs in {time.time()-t0:.2f}s")
            t0 = time.time()
        self.t = t
        return self.support_vectors[:self.current_size].copy(), self.support_weights[:self.current_size], self.alpha_indices

    def SolveIncremental(self, t, selected_values, X: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike):
        """
        Pegasos algorithm for k=1, single class
        """
        #print(X,y)
        X = X.reshape((-1,self.dimensionality))
        if self.support_vectors is None:
            self.sv_map[selected_values[0]] = 0
            self.alpha_indices.append(selected_values[0])
            self.support_vectors = X[0].reshape((1,self.dimensionality)).copy()
            self.support_weights = numpy.array([1], dtype=self.dtype)
            self.support_targets = numpy.array([y[0]], dtype=self.dtype)

        k = self.support_targets[:self.current_size] * self.kernel(X, self.support_vectors[:self.current_size])
        lambda_final = self.gamma * (t + len(y)/2)
        function_value = 2 * numpy.dot(self.support_weights[:self.current_size], k) -  self.kernel.InnerKernel(X, X).reshape(-1,1) + self.bias
        #function_value = numpy.dot(self.support_weights[:self.current_size], (k).squeeze()) + self.bias
        target = y
        offset = y #if self.use_regression_target else 0
        error = target  * (function_value)
        #error = self._Error(
        #            k, 
        #            self.support_weights[:self.current_size], 
        #            1 if self.use_regression_target else self.gamma * (t+1), 
        #            1 if self.use_regression_target else y,
        #            y - self.bias if self.use_regression_target else 0,
        #        )
        gradient = -error # * self.gamma #y * function_value #/ (1 + numpy.exp(function_value))
        #print(error)
        if self.use_regression_target:
            pass
        else:
            pass #error = error.clip(-1,1)
        #print(numpy.dot(k, self.support_weights[:self.current_size])+self.bias,error, y,self.current_size)
        for i, selected_value in enumerate(selected_values):
            if self.use_regression_target:
                self.support_weights *= (1 - 1.0/(t+i+1))
            if ( error[i] > 1/lambda_final and not self.use_regression_target) or (abs(error[i]) > 1 and self.use_regression_target):
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
                if not self.use_regression_target:
                    self.support_weights[self.sv_index] += gradient[i]  #+ 0.5 * numpy.sign(gradient)
                else:
                    delta = gradient[i]
                    #print(delta)
                    # SVR derivation: https://arxiv.org/ftp/arxiv/papers/1706/1706.01833.pdf
                    self.support_weights[self.sv_index] -= delta
                    #self.bias -= delta
                    k2 = self.support_targets[:self.current_size] * self.kernel(X, self.support_vectors[:self.current_size])
                    function_value2 = 2 * numpy.dot(self.support_weights[:self.current_size], k2) -  self.kernel.InnerKernel(X, X).reshape(-1,1) + self.bias
                    print(function_value[i], delta, function_value2[i],  y[i], "bias:",self.bias)
        #if self.use_regression_target:
        #    self.support_weights /= (t+i+1)