from load_datasets import *
from PegasosSVC import rbf_kernel, PegasosSVC
from PegasosSVR import SvrPredictor
import time, numpy
import matplotlib.pyplot as plt
import matplotlib

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

def test_svr_curve():
    """
    From: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
    """
    X = numpy.sort(5 * numpy.random.rand(40, 1), axis=0)
    y = numpy.sin(X).ravel()

    # add noise to targets
    y[::5] += 3 * (0.5 - numpy.random.rand(8))

    svr_rbf = SvrPredictor(C=100, kernel=rbf_kernel(gamma=0.1))
    #svr_lin = SVR(kernel="linear", C=100, gamma="auto")
    #svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

    lw = 2

    svrs = [svr_rbf] #, svr_lin, svr_poly]
    kernel_label = ["RBF", "Linear", "Polynomial"]
    model_color = ["m", "c", "g"]
    matplotlib.use("Agg")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        axes[ix].plot(
            X,
            svr.fit(X, y).predict(X),
            color=model_color[ix],
            lw=lw,
            label="{} model".format(kernel_label[ix]),
        )
        axes[ix].scatter(
            X[svr.indices],
            y[svr.indices],
            facecolor="none",
            edgecolor=model_color[ix],
            s=50,
            label="{} support vectors".format(kernel_label[ix]),
        )
        axes[ix].scatter(
            X[numpy.setdiff1d(numpy.arange(len(X)), svr.indices)],
            y[numpy.setdiff1d(numpy.arange(len(X)), svr.indices)],
            facecolor="none",
            edgecolor="k",
            s=50,
            label="other training data",
        )
        axes[ix].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=1,
            fancybox=True,
            shadow=True,
        )

    fig.text(0.5, 0.04, "data", ha="center", va="center")
    fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.savefig("out.png")

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