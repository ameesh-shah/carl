import sys
import glob
import tensorflow as tf

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    W = []
    for e in tf.train.summary_iterator(file):
        # print(e)
        for v in e.summary.value:
            if v.tag == 'mean-TRAIN-return':
                X.append(v.simple_value)
            elif v.tag == 'max-TRAIN-return':
                Y.append(v.simple_value)
            elif v.tag == 'TRAIN-mean-loss':
                Z.append(v.simple_value)
            elif v.tag == 'TRAIN-catastrophe-loss':
                W.append(v.simple_value)
    return X, Y, Z, W

if __name__ == '__main__':

    logdir = sys.argv[1] + "/events*"
    eventfile = glob.glob(logdir)[0]

    print(logdir)
    print(eventfile)

    X, Y, Z, W = get_section_results(eventfile)
    for i, (x, y, z, w) in enumerate(zip(X, Y, Z, W)):
        print('Iteration {:d} | mean Train return: {:f} | max Train return: {:f} | Train mean loss: {:f} | Train catastrophe loss: {:f}'.format(i, x, y, z, w))
