import sys
sys.path.append('/root/work/caffe/python/')
sys.path.append('/root/work/fcn.berkeleyvision.org/')

import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)
val = np.loadtxt('../data/pascal/VOCdevkit/MYVOC2012/ImageSets/Segmentation/val.txt', dtype=str)

for _ in range(25):
    solver.step(1)
    score.seg_tests(solver, False, val, layer='score')
