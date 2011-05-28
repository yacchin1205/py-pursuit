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

FLAGS.add_option('-b', '--batch-size', type=int, default=1)
FLAGS.add_option('-f', '--frames', action='store_true')
FLAGS.add_option('-m', '--min-coeff', type=float, default=0.1)
FLAGS.add_option('-x', '--max-num-coeffs', type=int, default=-1)
FLAGS.add_option('-a', '--alpha', type=float, default=0.1)
FLAGS.add_option('-t', '--tau', type=float, default=1.)
FLAGS.add_option('-r', '--rows', type=int, default=10)
FLAGS.add_option('-c', '--cols', type=int, default=10)

FLAGS.add_option('', '--l1', type=float, default=0.)
FLAGS.add_option('', '--l2', type=float, default=0.)
FLAGS.add_option('', '--momentum', type=float, default=0.)
FLAGS.add_option('', '--padding', type=float, default=0.1)
FLAGS.add_option('', '--grow', type=float, default=0.1)
FLAGS.add_option('', '--shrink', type=float, default=0.01)


def now():
    return datetime.datetime.now()


def load_image(path):
    return numpy.asarray(Image.open(path).convert('L')) / 128. - 1.


def save_frame(width, height):
    pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    Image.fromstring(mode="RGB", size=(width, height), data=pixels
                     ).transpose(Image.FLIP_TOP_BOTTOM
                                 ).save(now().strftime('frame-%Y%m%d-%H%M%S.%f.png'))


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
        arg = args[rng.randint(len(args))]
        devset.append(load_image(arg))
        args.remove(arg)
    images = [load_image(arg) for arg in args]
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

    width = max(x.shape[0] for x in images)
    height = max(x.shape[1] for x in images)

    source = numpy.zeros((width, height), 'f')
    reconst = numpy.zeros((width, height), 'f')
    filters = numpy.zeros((opts.rows, opts.cols, 20, 20), 'f')
    features = numpy.zeros((opts.rows, opts.cols, width, height), 'f')

    kwargs = dict(cmap=glumpy.colormap.Grey, vmin=-1, vmax=1)
    _source = glumpy.Image(source, **kwargs)
    _reconst = glumpy.Image(reconst, **kwargs)
    _filters = [[glumpy.Image(f, vmin=-0.2, vmax=0.2) for f in fs] for fs in filters]
    _features = [[glumpy.Image(f, vmin=0, vmax=3) for f in fs] for fs in features]

    W = 60 * (2 * opts.cols + 1) + 4
    H = 60 * opts.rows + 4

    win = glumpy.Window(W, H)

    def update():
        global batches, filters, _filters

        batches += 1
        batch = [images[rng.randint(len(images))] for _ in range(opts.batch_size)]

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

        enc = codebook.encode(pixels.copy(),
                              max_num_coeffs=opts.max_num_coeffs,
                              min_coeff=opts.min_coeff)

        estimate = codebook.decode(enc, pixels.shape)

        errors.append(abs(pixels - estimate).sum())
        logging.error('%d<%.3g>: error %d',
                      batches,
                      opts.alpha * opts.tau ** batches,
                      sum(errors) / max(1, len(errors)))

        w, h = pixels.shape

        source[:] = 0.
        source[:w, :h] = pixels
        _source.update()

        reconst[:] = 0.
        reconst[:w, :h] = estimate
        _reconst.update()

        features[:] = 0
        for i, c, (x, y) in enc:
            w, h = codebook.filters[i].shape
            a, b = numpy.unravel_index(i, (opts.rows, opts.cols))
            features[a, b, x:x+w, y:y+h] += c
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
        logging.error('%d<%.3g>: error %d',
                      batches,
                      opts.alpha * opts.tau ** batches,
                      sum(errors) / max(1, len(errors)))

    @win.event
    def on_draw():
        win.clear()
        W, H = win.get_size()
        p = 4
        w = int(float(W - p) / (2 * opts.cols))
        h = int(float(H - p) / (2 * opts.rows))
        _source.blit(p, h * opts.rows + p, w * opts.cols - p, h * opts.rows - p)
        _reconst.blit(w * opts.cols + p, h * opts.rows + p, w * opts.cols - p, h * opts.rows - p)
        for r in range(opts.rows):
            fs = _filters[r]
            Fs = _features[r]
            for c in range(opts.cols):
                fs[c].blit(w * c + p, h * r + p, w - p, h - p)
                Fs[c].blit(w * (c + opts.cols) + p, h * r + p, w - p, h - p)

    @win.event
    def on_idle(dt):
        global updates
        win.draw()
        if updates != 0:
            updates -= 1
            update()
            if opts.frames:
                save_frame(W, H)

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
