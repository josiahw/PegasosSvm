import pickle, numpy
import sklearn.datasets, sklearn.utils, sklearn.model_selection
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_mnist(train=12500,test=7500):
        # Load data from https://www.openml.org/d/554
        X, y = sklearn.datasets.fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')

        # set random state and permute data
        random_state = sklearn.utils.check_random_state(0)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        X = X.reshape((X.shape[0], -1)).astype(numpy.float32)

        # 60k samples in train set
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, train_size=train, test_size=test)
        del X, y
        return (X_train, X_test, y_train, y_test)

def load_moons(pts=10000, noise = 0.05):
    data, labels = sklearn.datasets.make_moons(pts,noise=noise,random_state=0)
    data -= numpy.mean(data,axis=0)
    # fix for svm
    labels[labels == 0] = -1
    return data, labels

class Cifar10:

    size = 50000
    train_data = (
        "datasets/cifar-10-batches-py/data_batch_1",
        "datasets/cifar-10-batches-py/data_batch_2",
        "datasets/cifar-10-batches-py/data_batch_3",
        "datasets/cifar-10-batches-py/data_batch_4",
        "datasets/cifar-10-batches-py/data_batch_5"
        )
    test_data = ["datasets/cifar-10-batches-py/test_batch"]

    def __init__(self):
        pass

    def size(self):
        return self.size

    def get_next(self):
        return None
    
    def get_all(self):
        data = unpickle(self.train_data[0])
        X = data[b'data']
        y = data[b'labels']
        #for d in self.train_data[1:]:
        #    data = unpickle(d)
        #    X = numpy.concatenate([X, data[b'data']])
        #    y = numpy.concatenate([y, data[b'labels']])
        return numpy.array(X, dtype=numpy.float32), numpy.array(y, dtype=numpy.float32)
    
    def get_test(self):
        data = unpickle(self.test_data[0])
        X = data[b'data']
        y = data[b'labels']
        if len(self.test_data) == 1:
            return numpy.array(X).squeeze().astype(numpy.float32), numpy.array(y).squeeze().astype(numpy.float32)
        for d in self.test_data[1:]:
            data = unpickle(d)
            X = numpy.concatenate([X, data[b'data']])
            y = numpy.concatenate([y, data[b'labels']])
        return X.astype(numpy.float32), y.astype(numpy.float32)

class Cifar100:

    def __init__(self):
        self.train_data = ["datasets/cifar-100-python/train"]
        self.test_data = ["datasets/cifar-100-python/test"]