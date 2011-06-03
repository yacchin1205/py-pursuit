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

FLAGS.add_option('-c', '--min-coeff', type=float, default=0.0001)
FLAGS.add_option('-n', '--max-num-coeffs', type=int, default=-1)
FLAGS.add_option('-a', '--min-activity-ratio', type=float, default=0.)

FLAGS.add_option('-R', '--rows', type=int, default=10)
FLAGS.add_option('-C', '--cols', type=int, default=10)

FLAGS.add_option('', '--learning-rate', type=float, default=0.1)
FLAGS.add_option('', '--padding', type=float, default=0.1)
FLAGS.add_option('', '--grow', type=float, default=0.01)
FLAGS.add_option('', '--shrink', type=float, default=0.0001)

FLAGS.add_option('-s', '--save-frames', action='store_true')


def now():
    return datetime.datetime.now()


def load_image(path):
    return (numpy.asarray(Image.open(path).convert('L')) - 128.) / 256.


def save_frame(width, height):
    pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    Image.fromstring(mode="RGB", size=(width, height), data=pixels
                     ).transpose(Image.FLIP_TOP_BOTTOM
                                 ).save(now().strftime('frame-%Y%m%d-%H%M%S.%f.png'))


class Simulator(object):
    def __init__(self, opts, args):
        self.opts = opts
        self.updates = -1
        self.errors = collections.deque(maxlen=10)

        self.devset = []
        for _ in range(min(len(args) // 2, 10)):
            arg = args[rng.randint(len(args))]
            self.devset.append(load_image(arg))
            args.remove(arg)
        self.images = [load_image(arg) for arg in args]

        self.codebook = lmj.pursuit.SpatialCodebook(
            opts.rows * opts.cols, (20, 20))

        self.trainer = lmj.pursuit.SpatialTrainer(
            self.codebook,
            min_coeff=opts.min_coeff,
            max_num_coeffs=opts.max_num_coeffs,
            min_activity_ratio=opts.min_activity_ratio,
            padding=opts.padding,
            grow=opts.grow,
            shrink=opts.shrink,
            )

        w = max(x.shape[0] for x in self.images)
        h = max(x.shape[1] for x in self.images)
        self.source = numpy.zeros((w, h), 'f')
        self.reconst = numpy.zeros((w, h), 'f')
        self.filters = numpy.zeros((opts.rows, opts.cols, 20, 20), 'f')
        self.features = numpy.zeros((opts.rows, opts.cols, w, h), 'f')

        kwargs = dict(cmap=glumpy.colormap.Grey, vmin=-0.5, vmax=0.5)
        self.source_image = glumpy.Image(self.source, **kwargs)
        self.reconst_image = glumpy.Image(self.reconst, **kwargs)
        self.filter_images = [
            [glumpy.Image(f, vmin=-0.2, vmax=0.2) for f in fs]
            for fs in self.filters]
        self.feature_images = [
            [glumpy.Image(f, vmin=0, vmax=3) for f in fs]
            for fs in self.features]

        self.iterator = self.learn_forever()

    def learn(self):
        self.source[:] = 0.
        self.reconst[:] = 0.
        self.features[:] = 0.

        pixels = self.images[rng.randint(len(self.images))].copy()
        w, h = pixels.shape
        self.source[:w, :h] += pixels

        grad = [numpy.zeros_like(w) for w in self.codebook.filters]
        activity = numpy.zeros((len(grad), ), float)

        for index, coeff, (x, y) in self.codebook.iterencode(
                pixels,
                self.opts.min_coeff,
                self.opts.max_num_coeffs):
            w, h = self.codebook.filters[index].shape[:2]

            grad[index] += coeff * pixels[x:x+w, y:y+h]
            activity[index] += coeff

            a, b = numpy.unravel_index(index, (opts.rows, opts.cols))
            self.features[a, b, x:x+w, y:y+h] += coeff

            self.reconst[x:x+w, y:y+h] += coeff * self.codebook.filters[index]

            yield

        self.trainer.apply_gradient(grad, activity, self.opts.learning_rate)

        w = max(w.shape[0] for w in self.codebook.filters)
        h = max(w.shape[1] for w in self.codebook.filters)

        if w > self.filters.shape[2] or h > self.filters.shape[3]:
            x = max(w * 1.5, h * 1.5)
            self.filters = numpy.zeros((self.opts.rows, self.opts.cols, x, x), 'f')
            self.filter_images = [
                [glumpy.Image(f, vmin=-0.2, vmax=0.2) for f in fs]
                for fs in self.filters]

        self.filters[:] = 0.
        for r in range(self.opts.rows):
            for c in range(self.opts.cols):
                f = self.codebook.filters[r * self.opts.cols + c]
                x, y = f.shape
                a = (self.filters.shape[2] - x) // 2
                b = (self.filters.shape[3] - y) // 2
                self.filters[r, c, a:a+x, b:b+y] += f

        [[f.update() for f in fs] for fs in self.filter_images]

    def learn_forever(self):
        while True:
            for _ in self.learn():
                yield
            self.errors.append(abs(self.source - self.reconst).sum())
            logging.error('error %d', sum(self.errors) / max(1, len(self.errors)))

    def draw(self, W, H, p=4):
        self.source_image.update()
        self.reconst_image.update()

        [[f.update() for f in fs] for fs in self.feature_images]

        w = int(float(W - p) / (2 * self.opts.cols))
        h = int(float(H - p) / (2 * self.opts.rows))
        self.source_image.blit(
            p, h * self.opts.rows + p, w * self.opts.cols - p, h * self.opts.rows - p)
        self.reconst_image.blit(w * self.opts.cols + p,
                                h * self.opts.rows + p,
                                w * self.opts.cols - p,
                                h * self.opts.rows - p)
        for r in range(self.opts.rows):
            fs = self.filter_images[r]
            Fs = self.feature_images[r]
            for c in range(self.opts.cols):
                fs[c].blit(w * c + p, h * r + p, w - p, h - p)
                Fs[c].blit(w * (c + self.opts.cols) + p, h * r + p, w - p, h - p)


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(levelname).1s %(asctime)s [%(module)s:%(lineno)d] %(message)s')

    numpy.seterr(all='raise')

    opts, args = FLAGS.parse_args()

    assert len(args) > 1, 'Usage: image.py FILE...'

    win = glumpy.Window()
    sim = Simulator(opts, args)

    @win.event
    def on_draw():
        win.clear()
        sim.draw(*win.get_size())

    @win.event
    def on_idle(dt):
        win.draw()
        if sim.updates != 0:
            sim.updates -= 1
            next(sim.iterator)
            if opts.save_frames:
                save_frame(*win.get_size())

    @win.event
    def on_key_press(key, modifiers):
        if key == glumpy.key.ESCAPE:
            sys.exit()
        if key == glumpy.key.SPACE:
            sim.updates = sim.updates == 0 and -1 or 0
        if key == glumpy.key.ENTER:
            if sim.updates >= 0:
                sim.updates = 1

    win.mainloop()
