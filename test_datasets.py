from load_datasets import *
from PegasosSVC import rbf_kernel, PegasosSVC
import time, numpy

def test_mnist(C = 3, gamma = 0.000001):
    X_train, X_test, y_train, y_test = load_mnist()
    clss = PegasosSVC(C,numpy.float32,rbf_kernel(gamma))
    t0 = time.time()
    clss.fit(X_train, y_train)
    print(f"fit in {time.time()-t0:.2f} seconds")

    #check assigned classes for the two moons as a classification error
    t0 = time.time()
    t = clss.predict(X_test)
    print(f"predicted in {time.time()-t0:.2f} seconds")
    print(t)
    error = numpy.sum((y_test!=t)**2) / float(len(y_test))
    return error

def test_cifar10(C = 1, gamma = 0.0000001):
    dataset = Cifar10()
    X_train, y_train= dataset.get_all()
    clss = PegasosSVC(C,numpy.float32,rbf_kernel(gamma))
    t0 = time.time()
    clss.fit(X_train, y_train)
    print(f"fit in {time.time()-t0:.2f} seconds")
    del X_train,y_train
    #check assigned classes for the two moons as a classification error
    X_test , y_test = dataset.get_test()
    t0 = time.time()
    t = clss.predict(X_test)
    print(f"predicted in {time.time()-t0:.2f} seconds")
    print(t)
    error = numpy.sum((y_test!=t)**2) / float(len(y_test))
    return error