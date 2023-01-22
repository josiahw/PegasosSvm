from matplotlib import pyplot

def plot_svc_moons(clss)
    #generate a heatmap and display classified clusters.
    matplotlib.use("Agg")
    a = numpy.zeros((100,100))
    for i in range(100):
        for j in range(100):
            a[j,i] = clss.predict_proba(numpy.array([i*4/100.-2,j*4/100.-2]))[0][0]
    pyplot.imshow(a, cmap='hot', interpolation='nearest')
    data *= 25.
    data += 50.
    pyplot.scatter(data[t==0,0],data[t==0,1],c='r')
    pyplot.scatter(data[t==1,0],data[t==1,1],c='b')
    pyplot.savefig("out.png")

def plot_svr_curve(y, y_pred):
    #generate a heatmap and display classified clusters.
    matplotlib.use("Agg")
    a = numpy.zeros((100,100))
    for i in range(100):
        for j in range(100):
            a[j,i] = clss.predict_proba(numpy.array([i*4/100.-2,j*4/100.-2]))[0][0]
    pyplot.imshow(a, cmap='hot', interpolation='nearest')
    data *= 25.
    data += 50.
    pyplot.scatter(data[t==0,0],data[t==0,1],c='r')
    pyplot.scatter(data[t==1,0],data[t==1,1],c='b')
    pyplot.savefig("out.png")

def plot_confusion(y, y_pred):
    #generate a heatmap and display classified clusters.
    matplotlib.use("Agg")
    a = numpy.zeros((100,100))
    for i in range(100):
        for j in range(100):
            a[j,i] = clss.predict_proba(numpy.array([i*4/100.-2,j*4/100.-2]))[0][0]
    pyplot.imshow(a, cmap='hot', interpolation='nearest')
    data *= 25.
    data += 50.
    pyplot.scatter(data[t==0,0],data[t==0,1],c='r')
    pyplot.scatter(data[t==1,0],data[t==1,1],c='b')
    pyplot.savefig("out.png")