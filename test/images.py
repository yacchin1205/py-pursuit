#!/usr/bin/env python

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

import collections
import glumpy
import logging
import numpy
import numpy.random as rng
import OpenGL
import optparse
import os
import PIL.Image
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import lmj.pursuit

FLAGS = optparse.OptionParser('test/images.py [options] FILE...')
FLAGS.add_option('-c', '--min-coeff', type=float, default=5)
FLAGS.add_option('-d', '--min-coeff-decay', type=float, default=0.9)
FLAGS.add_option('-n', '--max-num-coeffs', type=int, default=-1)
FLAGS.add_option('-f', '--filters', type=int, default=7)
FLAGS.add_option('-F', '--filter-size', type=int, default=16)
FLAGS.add_option('-r', '--learning-rate', type=float, default=0.05)

FLAGS.add_option('-s', '--save-frames', type=int, default=0)

FLAGS.add_option('', '--min-activity-ratio', type=float, default=0.1)
FLAGS.add_option('', '--activity-recency', type=float, default=0.9)
FLAGS.add_option('', '--padding', type=float, default=0.1)
FLAGS.add_option('', '--grow', type=float, default=0.2)
FLAGS.add_option('', '--shrink', type=float, default=0.02)

FLAGS.add_option('', '--model')


def load_image(path):
    a = numpy.asarray(PIL.Image.open(path).convert('L'), float)
    a -= a.mean()
    a /= a.std()
    return a


def save_frame(i, width, height):
    logging.info('%d: saving %dx%d frame', i, width, height)
    img = PIL.Image.fromstring(
        mode='RGB',
        size=(width, height),
        data=OpenGL.GL.glReadPixels(
            0, 0, width, height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE))
    img.transpose(PIL.Image.FLIP_TOP_BOTTOM).save('frame-%016d.png' % i)


WhiteBlack = glumpy.colormap.Colormap(
    'WhiteBlack',
    (0.00, (1.0, 1.0, 1.0)),
    (1.00, (0.0, 0.0, 0.0)))


class Simulator(object):
    def __init__(self, source_frame, target_frame, filter_frames, feature_frames, opts, args):
        self.opts = opts

        self.frames_until_pause = -1
        self.frames = 0
        self.frames_saved = 0

        # load images.

        self.devset = []
        for _ in range(min(len(args) // 2, 10)):
            arg = args[rng.randint(len(args))]
            self.devset.append(load_image(arg))
            args.remove(arg)
        self.images = [load_image(arg) for arg in args]

        # set up a matching pursuit object to train.

        F = opts.filters
        S = opts.filter_size
        self.codebook = lmj.pursuit.CorrelationCodebook(F * F, (S, S))
        self.trainer = lmj.pursuit.CorrelationTrainer(self.codebook)
        self.activity = numpy.zeros((F * F, ), float)

        self.learner = self.learn_forever()
        self.errors = collections.deque(maxlen=10)
        self.min_coeff = opts.min_coeff

        # set up glumpy image objects to display what's happening on the screen.

        w = max(x.shape[0] for x in self.images)
        h = max(x.shape[1] for x in self.images)
        vs = sorted(numpy.concatenate([abs(f).flatten() for f in self.codebook.filters]))
        v = vs[int(0.9 * len(vs))]

        kws = dict(vmin=-2, vmax=2, colormap=glumpy.colormap.Grey)
        self.source = numpy.zeros((w, h), 'f')
        self.source_image = glumpy.image.Image(self.source, **kws)
        self.source_frame = source_frame

        self.target = numpy.zeros((w, h), 'f')
        self.target_image = glumpy.image.Image(self.target, **kws)
        self.target_frame = target_frame

        kws = dict(vmin=0, vmax=25, colormap=glumpy.colormap.IceAndFire)
        self.features = numpy.zeros((F, F, w, h), 'f')
        self.feature_images = [[glumpy.image.Image(f, **kws) for f in fs] for fs in self.features]
        self.feature_frames = feature_frames

        kws['vmin'] = -v
        kws['vmax'] = v
        self.filters = numpy.zeros((F, F, S, S), 'f')
        self.filter_images = [[glumpy.image.Image(f, **kws) for f in fs] for fs in self.filters]
        self.filter_frames = filter_frames

    def learn_forever(self):
        while True:
            self.refresh_filters()
            for _ in self.learn():
                yield
            err = numpy.linalg.norm(self.source - self.target)
            self.errors.append(err / numpy.sqrt(self.source.size))
            logging.error('rmse %.2f', sum(self.errors) / max(1, len(self.errors)))

    def learn(self):
        self.source[:] = 0.
        self.target[:] = 0.
        self.features[:] = 0.

        pixels = self.images[rng.randint(len(self.images))].copy()
        w, h = pixels.shape[:2]
        self.source[:w, :h] += pixels

        grad = [numpy.zeros_like(f) for f in self.codebook.filters]
        activity = numpy.zeros_like(self.activity)
        encoding = []
        for index, coeff, (x, y) in self.codebook.encode(
                pixels,
                min_coeff=self.min_coeff,
                max_num_coeffs=self.opts.max_num_coeffs):
            encoding.append((index, coeff, x, y))
            a, b = numpy.unravel_index(index, (opts.filters, opts.filters))
            w, h = self.codebook.filters[index].shape[:2]
            self.features[a, b, x:x+w, y:y+h] += 10 * abs(rng.randn()) # coeff
            self.target[x:x+w, y:y+h] += coeff * self.codebook.filters[index]
            grad[index] += coeff * pixels[x:x+w, y:y+h]
            activity[index] += coeff
            yield
            self.frames += 1

        #error = self.source - self.target
        #for index, coeff, x, y in encoding:
        #    w, h = self.codebook.filters[index].shape[:2]
        #    grad[index] += coeff * error[x:x+w, y:y+h]
        #    activity[index] += coeff

        self.activity *= 1 - self.opts.activity_recency
        self.activity += self.opts.activity_recency * activity

        self.trainer.apply_gradient(
            (g / (a or 1) for g, a in zip(grad, activity)), self.opts.learning_rate)
        self.trainer.resize(self.opts.padding, self.opts.shrink, self.opts.grow)
        self.trainer.resample(self.activity, self.opts.min_activity_ratio)

        self.min_coeff *= self.opts.min_coeff_decay

    def refresh_filters(self):
        w = max(f.shape[0] for f in self.codebook.filters)
        h = max(f.shape[1] for f in self.codebook.filters)
        vs = sorted(numpy.concatenate([abs(f).flatten() for f in self.codebook.filters]))
        v = vs[int(0.9 * len(vs))]
        if w > self.filters.shape[2] or h > self.filters.shape[3]:
            shape = list(self.filters.shape)
            shape[2] = shape[3] = max(w * 1.2, h * 1.2)
            kws = dict(vmin=-v, vmax=v)
            self.filters = numpy.zeros(shape, 'f')
            self.filter_images = [[glumpy.Image(f, **kws) for f in fs] for fs in self.filters]

        self.filters[:] = 0.
        for r in xrange(self.opts.filters):
            for c in xrange(self.opts.filters):
                f = self.codebook.filters[r * self.opts.filters + c]
                x, y = f.shape[:2]
                a = (self.filters.shape[2] - x) // 2
                b = (self.filters.shape[3] - y) // 2
                self.filters[r, c, a:a+x, b:b+y] = f

    def draw(self):
        def render(i, f):
            i.update()
            f.draw(x=f.x, y=f.y)
            i.draw(x=f.x, y=f.y, z=0, width=f.width, height=f.height)

        render(self.source_image, self.source_frame)
        render(self.target_image, self.target_frame)
        for r in xrange(self.opts.filters):
            for c in xrange(self.opts.filters):
                render(self.filter_images[r][c], self.filter_frames[r][c])
                render(self.feature_images[r][c], self.feature_frames[r][c])


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(levelname).1s %(asctime)s [%(module)s:%(lineno)d] %(message)s')

    numpy.seterr(all='raise')

    opts, args = FLAGS.parse_args()

    if not args:
        FLAGS.error('No images specified!')

    def add(r, c, size=(1, 1), spacing=0.025):
        f = fig.add_figure(cols=N, rows=N, position=(r, c), size=size)
        return f.add_frame(spacing=spacing)

    n = opts.filters
    N = 2 * n
    fig = glumpy.figure(size=(800, 800))
    sim = Simulator(
        add(0, n, size=(n, n), spacing=0.025 / n),
        add(n, n, size=(n, n), spacing=0.025 / n),
        [[add(r, c) for c in xrange(n)] for r in xrange(n)],
        [[add(n + r, c) for c in xrange(n)] for r in xrange(n)],
        opts, args)

    @fig.event
    def on_draw():
        fig.clear()
        sim.draw()

    @fig.event
    def on_idle(dt):
        if sim.frames_until_pause != 0:
            sim.frames_until_pause -= 1
            try:
                sim.learner.next()
            except:
                logging.exception('error while training !')
                sys.exit()
            if opts.save_frames and 0 == sim.frames % opts.save_frames:
                save_frame(sim.frames_saved, fig.width, fig.height)
                sim.frames_saved += 1
        fig.redraw()

    @fig.event
    def on_key_press(key, modifiers):
        if key == glumpy.window.key.ESCAPE:
            sys.exit()
        if key == glumpy.window.key.SPACE:
            sim.frames_until_pause = sim.frames_until_pause == 0 and -1 or 0
        if key == glumpy.window.key.ENTER:
            if sim.frames_until_pause >= 0:
                sim.frames_until_pause = 1

    glumpy.show()
