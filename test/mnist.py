# Copyright (c) 2011 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import time
import numpy
import glumpy
import pickle
import logging
import datetime
import optparse
import collections
import numpy.random as rng

from PIL import Image
from OpenGL import GL as gl

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import lmj.pursuit

FLAGS = optparse.OptionParser()
FLAGS.add_option('', '--model')

FLAGS.add_option('-b', '--batch-size', type=int, default=10)
FLAGS.add_option('-f', '--frames', action='store_true')
FLAGS.add_option('-m', '--min-coeff', type=float, default=0.1)
FLAGS.add_option('-x', '--max-num-coeffs', type=int, default=1000)
FLAGS.add_option('-a', '--alpha', type=float, default=0.1)
FLAGS.add_option('-t', '--tau', type=float, default=0.999)
FLAGS.add_option('-r', '--rows', type=int, default=5)
FLAGS.add_option('-c', '--cols', type=int, default=5)
FLAGS.add_option('-W', '--width', type=int, default=400)
FLAGS.add_option('-H', '--height', type=int, default=400)

FLAGS.add_option('', '--l1', type=float, default=0.)
FLAGS.add_option('', '--l2', type=float, default=0.)
FLAGS.add_option('', '--momentum', type=float, default=0.)
FLAGS.add_option('', '--padding', type=float, default=0.1)
FLAGS.add_option('', '--grow', type=float, default=0.15)
FLAGS.add_option('', '--shrink', type=float, default=0.015)


def now():
    return datetime.datetime.now()


def save_frame(width, height):
    pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    Image.fromstring(mode="RGB", size=(width, height), data=pixels
                     ).transpose(Image.FLIP_TOP_BOTTOM
                                 ).save(now().strftime('frame-%Y%m%d-%H%M%S.%f.png'))


def load_image(path):
    a = numpy.asarray(Image.open(path).convert('L'))
    return (a - 128.) / 128.


def mean(x):
    return sum(x) / max(1, len(x))


def choose(a):
    '''Get a random element from sequence a.'''
    return a[rng.randint(0, len(a) - 1)]


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(levelname).1s %(asctime)s [%(module)s:%(lineno)d] %(message)s')

    numpy.seterr(all='raise')

    opts, args = FLAGS.parse_args()

    assert len(args) > 1, 'Usage: image.py FILE...'

    updates = -1
    batches = 0.
    devset = []
    for _ in range(min(len(args) // 2, 10)):
        arg = choose(args)
        devset.append(load_image(arg))
        args.remove(arg)
    trainset = [load_image(arg) for arg in args]
    errors = collections.deque(maxlen=10)

    codebook = (opts.model and
            pickle.load(open(opts.model, 'rb')) or
            lmj.pursuit.SpatialCodebook(opts.rows * opts.cols, (10, 10)))

    trainer = lmj.pursuit.SpatialTrainer(
        codebook,
        min_coeff=opts.min_coeff,
        max_num_coeffs=opts.max_num_coeffs,
        momentum=opts.momentum,
        l1=opts.l1,
        l2=opts.l2,
        padding=opts.padding,
        grow=opts.grow,
        shrink=opts.shrink,
        )

    width = max(x.shape[0] for x in trainset)
    height = max(x.shape[1] for x in trainset)

    source = numpy.zeros((width, height), 'f')
    reconst = numpy.zeros((width, height), 'f')
    filters = numpy.zeros((opts.rows, opts.cols, 20, 20), 'f')
    features = numpy.zeros((opts.rows, opts.cols, width, height), 'f')

    kwargs = dict(cmap=glumpy.colormap.Grey, vmin=0, vmax=255)
    _source = glumpy.Image(source, **kwargs)
    _reconst = glumpy.Image(reconst, **kwargs)
    _filters = [[glumpy.Image(f, vmin=-0.3, vmax=0.3) for f in fs] for fs in filters]
    _features = [[glumpy.Image(f, vmin=0, vmax=2) for f in fs] for fs in features]

    W = 60 * (2 * opts.cols + 1) + 4
    H = 60 * opts.rows + 4

    win = glumpy.Window(W, H)

    def learn():
        global batches
        batch = [choose(trainset) for _ in range(opts.batch_size)]
        batches += 1

        start = time.time()
        Grad = None
        for pixels in batch:
            #trainer.learn(pixels, opts.alpha)
            grad = trainer.calculate_gradient(pixels.copy())
            if Grad is None:
                Grad = list(grad)
            else:
                for G, g in zip(Grad, grad):
                    G += g
        trainer.apply_gradient(
            (g / opts.batch_size for g in Grad),
            learning_rate=opts.alpha * opts.tau ** batches)
        logging.info('processed batch in %dms', 1000 * (time.time() - start))

        return pixels

    def update(pixels):
        global filters, _filters

        enc = codebook.encode(pixels.copy(),
                              max_num_coeffs=opts.max_num_coeffs,
                              min_coeff=opts.min_coeff)

        w, h = pixels.shape

        source[:] = 0.
        source[:w, :h] = pixels * 128 + 128
        _source.update()

        reconst[:] = 0.
        reconst[:w, :h] = codebook.decode(enc, pixels.shape) * 128 + 128
        _reconst.update()

        features[:] = 0
        for i, c, o in enc:
            features[numpy.unravel_index(i, (opts.rows, opts.cols)) + o] += c
        [[f.update() for f in fs] for fs in _features]

        # re-allocate new filter arrays if filters have grown too much.
        width = max(f.shape[0] for f in codebook.filters)
        height = max(f.shape[0] for f in codebook.filters)
        if width > 1.1 * filters.shape[2] or height > 1.1 * filters.shape[3]:
            filters = numpy.zeros((opts.rows, opts.cols, 1.5 * width, 1.5 * height), 'f')
            _filters = [[glumpy.Image(f, vmin=-0.3, vmax=0.3) for f in fs] for fs in filters]

        filters[:] = 0
        for r in range(opts.rows):
            for c in range(opts.cols):
                f = codebook.filters[r * opts.cols + c]
                x, y = f.shape
                a = (filters.shape[2] - x) // 2
                b = (filters.shape[3] - y) // 2
                filters[r, c, a:a+x, b:b+y] += f
        [[f.update() for f in fs] for fs in _filters]

    def evaluate():
        for t, pixels in enumerate(devset):
            estimate = trainer.reconstruct(pixels.copy())
            errors.append(abs(pixels - estimate).sum())
        logging.error('%d<%.3g>: error %d', batches, opts.alpha * opts.tau ** batches, mean(errors))

    @win.event
    def on_draw():
        win.clear()
        p = 4
        W, H = win.get_size()
        w = int(float(W - p) / (2 * opts.cols + 1))
        h = int(float(H - p) / opts.rows)
        _source.blit(w * opts.cols + p, p, w - p, h - p)
        _reconst.blit(w * opts.cols + p, h + p, w - p, h - p)
        for r in range(opts.rows):
            fs = _filters[r]
            Fs = _features[r]
            for c in range(opts.cols):
                fs[c].blit(w * c + p, h * r + p, w - p, h - p)
                Fs[c].blit(w * (c + 1 + opts.cols) + p, h * r + p, w - p, h - p)

    @win.event
    def on_idle(dt):
        global updates
        if updates != 0:
            updates -= 1
            update(learn())
            win.draw()
            if opts.frames:
                save_frame(W, H)
            if rng.random() < 0.1:
                evaluate()

    @win.event
    def on_key_press(key, modifiers):
        global updates
        if key == glumpy.key.ESCAPE:
            sys.exit()
        if key == glumpy.key.S:
            pickle.dump(book, open(now().strftime('pursuit-%Y%m%d-%H%M%S.p'), 'wb'))
        if key == glumpy.key.SPACE:
            updates = updates == 0 and -1 or 0
        if key == glumpy.key.ENTER:
            if updates >= 0:
                updates = 1

    win.mainloop()
