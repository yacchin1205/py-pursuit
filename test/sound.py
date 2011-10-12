# Copyright (c) 2009 Leif Johnson <leif@leifjohnson.net>
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

'''This is a simple test for the Matching Pursuit sparse coding algorithm.'''

import os
import sys
import wave
import numpy
import logging
import numpy.random as rng

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import lmj.pursuit

SAMPLE_RATE = 44100

GRAPHS = ''
if GRAPHS:
    import scipy.signal
    import matplotlib.pylab as plt


def read_frames(filename):
    snd = wave.open(filename)
    assert snd.getframerate() == SAMPLE_RATE
    assert snd.getsampwidth() == 2
    assert snd.getnchannels() == 1

    data = numpy.fromstring(snd.readframes(-1), dtype=numpy.int16).astype(float)
    data -= data.mean()
    data /= abs(data).max()
    return data


def random_windows(n, samples, width):
    window = (0.5 * (1 - numpy.cos(2 * numpy.pi * numpy.arange(width) / (width - 1))))
    for _ in range(n):
        o = rng.randint(0, len(samples) - width - 1)
        yield window * samples[o:o+width]


def iter_windows(samples, width):
    o = 0
    window = (0.5 * (1 - numpy.cos(2 * numpy.pi * numpy.arange(width) / (width - 1))))
    while o + width <= len(samples):
        yield o, window * samples[o:o+width]
        o += int(0.5 * width)


def rmse(a, b):
    return '\t%.1f' % (1000 * numpy.linalg.norm(a - b) / numpy.sqrt(len(a)))


def evaluate(train, test, width, codebook=10):
    print codebook, '\t', width,

    def plot(label):
        if not GRAPHS:
            return
        for i, f in enumerate(c.codebook):
            plt.plot([i + 1] * len(f), 'k-', lw=0.5, alpha=0.5, aa=True)
            plt.plot(i + 1 + f, 'k-', lw=1, alpha=0.8, aa=True)
        o = rng.randint(0, len(train) - width - 1)
        plt.plot(train[o:o+width], 'r-', aa=True)
        plt.gca().set_yticks([])
        plt.box(False)
        plt.savefig(os.path.join(GRAPHS, '%s-w%03d-c%03d-%05d-basis.png' % (label, width, codebook, seq)))
        plt.clf()

    def plot_residual(label, a, b):
        if not GRAPHS:
            return
        win = 300
        corr = scipy.signal.correlate(abs(a), numpy.ones(win), 'valid')
        i = min(corr.argmax(), len(a) - win)
        plt.plot(a[i:i+win], 'b', alpha=0.8, aa=True)
        plt.plot(b[i:i+win], 'k', alpha=0.8, aa=True)
        plt.plot((a - b)[i:i+win], 'r', alpha=0.8, aa=True)
        plt.savefig(os.path.join(GRAPHS, '%s-w%03d-c%03d-%05d-resid.png' % (label, width, codebook, seq)))
        plt.clf()

    # windowed
    seq = 0
    c = lmj.pursuit.Codebook(codebook, width)
    t = lmj.pursuit.Trainer(c, max_num_coeffs=1)
    for _ in range(4):
        for w in random_windows(500, train, width):
            t.learn(w, 0.3)
        seq += 1
        plot('window')
        s = numpy.zeros_like(test)
        for o, w in iter_windows(test, width):
            s[o:o + width] = t.reconstruct(w)
        plot_residual('window', s, test)
        print rmse(s, test),
        sys.stdout.flush()

    # continuous
    seq = 0
    c = lmj.pursuit.correlation.Codebook(codebook, width)
    t = lmj.pursuit.correlation.Trainer(c, max_num_coeffs=500)
    for _ in range(4):
        _, activity = t.learn(train, 0.3)
        t.resize(0.1, 0.01, 0.001)
        t.resample(activity, 0.1)
        seq += 1
        plot('full')
        s = t.reconstruct(test)
        plot_residual('full', s, test)
        print rmse(s, test),
        sys.stdout.flush()

    print


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format='%(levelname).1s %(asctime)s %(message)s')

    w = rng.randint(int(SAMPLE_RATE * 0.5), int(SAMPLE_RATE * 1.5))

    train = read_frames(os.path.join(os.path.dirname(__file__), 'lullaby.wav'))
    a = rng.randint(0, len(train) - w)
    train = train[a:a+w]

    test = read_frames(os.path.join(os.path.dirname(__file__), 'lullaby.wav'))
    a = rng.randint(0, len(test) - w)
    test = test[a:a+w]

    rms = '%.1f' % (1000 * numpy.linalg.norm(test) / numpy.sqrt(len(test)))
    print 'Matching Pursuit Test'
    print 'We are encoding a test signal with rms power', rms
    print 'This table lists rms reconstruction error for different codebooks:'
    print '\t'.join('count length single . . . multiple'.split()).replace('.', '')
    for codebook in (10, 20, 50):
        for width in (0.1, 0.2, 0.5):
            evaluate(train, test, int(SAMPLE_RATE / 100. * width), codebook)
