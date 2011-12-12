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

import cairo
import collections
import gobject
import gtk
import logging
import math
import numpy
import numpy.random as rng
import optparse
import sys

import lmj.pursuit

TAU = 2 * numpy.pi

FLAGS = optparse.OptionParser()
FLAGS.add_option('-v', '--verbose', action='store_true', help='be more verbose')
FLAGS.add_option('-s', '--variance', type=float, default=0.1, help='generate gaussians with variance S')
FLAGS.add_option('-l', '--sample-history', type=int, default=1000, help='display the most recent N samples')


def dot(ctx, center, color=(0, 0, 0), size=2, alpha=1.0):
    '''Draw a dot at the given location in the given color.'''
    ctx.set_source_rgba(*(color + (alpha, )))
    ctx.arc(center[0], center[1], size, 0, 2 * math.pi)
    ctx.fill()


class Viewer(gtk.DrawingArea):
    __gsignals__ = {'expose-event': 'override'}

    def __init__(self, opts):
        super(Viewer, self).__init__()

        self.teacher = None
        self.iteration = 0

        n = 3
        self.means = []
        self.covariances = []
        for i in range(n):
            angle = numpy.array([numpy.cos(i * TAU / n), numpy.sin(i * TAU / n)])
            self.means.append(0.5 * rng.randn(2) + 2 * angle)
            cov = opts.variance * (numpy.eye(2) + 0.5 * rng.exponential(size=(2, 2)))
            cov[1, 0] = cov[0, 1]
            self.covariances.append(cov)

        self.codebook = lmj.pursuit.Codebook(2 * n, 2)
        self.trainer = lmj.pursuit.Trainer(self.codebook, max_num_coeffs=2)

        self.samples = collections.deque(maxlen=opts.sample_history)

    def sample(self):
        '''Generate one sample from our mixture.'''
        ws = numpy.zeros(len(self.means))
        ws[rng.randint(len(self.means))] = 1
        r = rng.multivariate_normal
        p = sum(w * r(self.means[i], self.covariances[i]) for i, w in enumerate(ws))
        self.samples.append((ws, p))
        return p

    def do_keypress(self, window, event):
        k = event.string or gtk.gdk.keyval_name(event.keyval).lower()

        if not k:
            return

        elif k == '\x1b':
            gtk.main_quit()

        elif k == ' ':
            if self.teacher:
                gobject.source_remove(self.teacher)
                self.teacher = None
            else:
                self.teacher = gobject.timeout_add(50, self.learn)

        self.queue_draw()

    def learn(self):
        p = self.sample()
        g, _ = self.trainer.calculate_gradient(p)
        self.trainer.apply_gradient(g, 0.1)
        self.iteration += 1
        self.queue_draw()
        return True

    def do_expose_event(self, event):
        ctx = self.window.cairo_create()
        ctx.rectangle(event.area.x,
                      event.area.y,
                      event.area.width,
                      event.area.height)
        ctx.clip()

        width, height = self.window.get_size()
        ctx.set_source_rgb(1, 1, 1)
        ctx.rectangle(0, 0, width, height)
        ctx.fill()

        ctx.set_source_rgb(0, 0, 0)
        ctx.move_to(10, 20)
        ctx.show_text('iteration %d' % self.iteration)

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.translate(width / 2, height / 2)
        z = min(width / 5.0, height / 5.0)
        ctx.scale(z, -z)

        self.draw_centers(ctx)
        self.draw_samples(ctx)
        self.draw_codebook(ctx)

    def draw_centers(self, ctx):
        color = collections.deque([1, 0, 0], maxlen=3)
        for i, mu in enumerate(self.means):
            ctx.set_source_rgba(*(tuple(color) + (0.1, )))
            color.appendleft(0)
            ctx.save()

            u, s, v = numpy.linalg.svd(self.covariances[i])

            ctx.translate(*mu)
            ctx.rotate(math.atan2(u[1, 0], u[0, 0]))
            ctx.scale(numpy.sqrt(s[0]), numpy.sqrt(s[1]))

            ctx.arc(0, 0, 1, 0, TAU)
            ctx.fill()

            ctx.restore()

    def draw_samples(self, ctx):
        for i, (ws, (x, y)) in enumerate(self.samples):
            ctx.set_source_rgba(*(tuple(ws) + (0.5, )))
            ctx.arc(x, y, i == len(self.samples) - 1 and 0.05 or 0.01, 0, TAU)
            ctx.fill()

    def draw_codebook(self, ctx):
        ctx.set_source_rgba(0, 0, 0, 0.5)
        ctx.set_line_width(0.01)
        for x, y in self.codebook.filters:
            ctx.move_to(0, 0)
            ctx.line_to(x, y)
            ctx.stroke()


if __name__ == '__main__':
    opts, args = FLAGS.parse_args()

    logging.basicConfig(stream=sys.stdout,
                        level=opts.verbose and logging.DEBUG or logging.INFO,
                        format='%(levelname).1s %(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    v = Viewer(opts)
    v.show()

    w = gtk.Window()
    w.set_title('Matching Pursuits')
    w.set_default_size(800, 600)
    w.add(v)
    w.present()
    w.connect('key-press-event', v.do_keypress)
    w.connect('delete-event', gtk.main_quit)

    gtk.main()


