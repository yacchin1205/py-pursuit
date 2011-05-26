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

import sys
import gzip
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

import lmj.pursuit

FLAGS = optparse.OptionParser()
FLAGS.add_option('', '--model')

FLAGS.add_option('-b', '--batch-size', type=int, default=20)
FLAGS.add_option('-f', '--frames', action='store_true')
FLAGS.add_option('-m', '--min-coeff', type=float, default=0.)
FLAGS.add_option('-x', '--max-num-coeffs', type=int, default=-1)
FLAGS.add_option('-a', '--alpha', type=float, default=0.5)
FLAGS.add_option('-t', '--tau', type=float, default=0.99)
FLAGS.add_option('-r', '--rows', type=int, default=5)
FLAGS.add_option('-c', '--cols', type=int, default=5)

FLAGS.add_option('', '--l1', type=float, default=0.)
FLAGS.add_option('', '--l2', type=float, default=0.)
FLAGS.add_option('', '--momentum', type=float, default=0.)
FLAGS.add_option('', '--padding', type=float, default=0.1)
FLAGS.add_option('', '--grow', type=float, default=0.05)
FLAGS.add_option('', '--shrink', type=float, default=0.005)


def now():
    return datetime.datetime.now()


def save_frame(width, height):
    pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    Image.fromstring(mode="RGB", size=(width, height), data=pixels
                     ).transpose(Image.FLIP_TOP_BOTTOM
                                 ).save(now().strftime('frame-%Y%m%d-%H%M%S.%f.png'))


def mean(x):
    return sum(x) / max(1, len(x))


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(levelname).1s %(asctime)s [%(module)s:%(lineno)d] %(message)s')

    opts, args = FLAGS.parse_args()

    assert len(args) == 1, 'Usage: pursuit_mnist.py DATAFILE'

    handle = gzip.open(args[0])
    dataset = pickle.load(handle)
    handle.close()

    devset = [dataset[i][0] for i in range(10)]
    updates = -1
    batches = 0.
    errors = [collections.deque(maxlen=20) for _ in range(10)]

    book = (opts.model and
            pickle.load(open(opts.model, 'rb')) or
            lmj.pursuit.SpatialCodebook(opts.rows * opts.cols, (10, 10)))

    trainer = lmj.pursuit.SpatialTrainer(
        book,
        min_coeff=opts.min_coeff,
        max_num_coeffs=opts.max_num_coeffs,
        momentum=opts.momentum,
        l1=opts.l1,
        l2=opts.l2,
        padding=opts.padding,
        grow=opts.grow,
        shrink=opts.shrink,
        )

    digit = numpy.zeros((28, 28), 'f')
    reconst = numpy.zeros((28, 28), 'f')
    filters = numpy.zeros((opts.rows, opts.cols, 28, 28), 'f')

    kwargs = dict(cmap=glumpy.colormap.Grey, vmin=0, vmax=255)
    _digit = glumpy.Image(digit, **kwargs)
    _reconst = glumpy.Image(reconst, **kwargs)
    _filters = [[glumpy.Image(f, vmin=-1, vmax=1) for f in fs] for fs in filters]

    W = 60 * (opts.cols + 1) + 4
    H = 60 * opts.rows + 4

    win = glumpy.Window(W, H)

    def get_pixels():
        label = rng.randint(0, 9)
        images = dataset[label]
        pixels = images[rng.randint(1, len(images) - 1)].astype(float)
        return (pixels - 128.) / 128.

    def learn():
        global batches
        batches += 1

        start = time.time()
        batch = numpy.array([get_pixels() for _ in range(opts.batch_size)])
        logging.info('loaded %d: %.3g/%.3g/%.3g in %dms', len(batch),
                     batch.min(), batch.mean(), batch.max(),
                     1000 * (time.time() - start))

        start = time.time()
        Grad = None
        for pixels in batch:
            #trainer.learn(pixels, opts.alpha)
            grad = trainer.calculate_gradient(pixels)
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
        digit[:] = pixels
        _digit.update()

        reconst[:] = trainer.reconstruct(pixels) * 128. + 128.
        _reconst.update()

        filters[:] = 0
        for r in range(opts.rows):
            for c in range(opts.cols):
                f = book.filters[r * opts.cols + c]
                filters[r, c, :f.shape[0], :f.shape[1]] += f
        [[f.update() for f in fs] for fs in _filters]

    def evaluate():
        for t, pixels in enumerate(devset):
            estimate = trainer.reconstruct(pixels)
            errors[t].append(((pixels - estimate) ** 2).mean())
        logging.error('%d<%.3g>: %s',
                      batches,
                      opts.alpha * opts.tau ** batches,
                      ' : '.join('%d' % mean(errors[t]) for t in range(10)))

    @win.event
    def on_draw():
        win.clear()
        _digit.blit(4, 4, 56, 56)
        _reconst.blit(4, 64, 56, 56)
        for r in range(opts.rows):
            fs = _filters[r]
            for c in range(opts.cols):
                fs[c].blit(60 * (c + 1) + 4, 60 * r + 4, 56, 56)

    @win.event
    def on_idle(dt):
        global updates
        if updates != 0:
            updates -= 1
            update(learn())
            win.draw()
            if opts.frames:
                save_frame(W, H)
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
