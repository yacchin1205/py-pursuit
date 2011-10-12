# Copyright (c) 2009 - 2011 Leif Johnson <leif@leifjohnson.net>
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

import numpy
import logging
import numpy.random as rng


class Codebook(object):
    '''Matching pursuit encodes signals using a codebook of filters.

    The encoding process decomposes a signal recursively into a maximally
    responding filter and a residual. Formally, the encoding process takes a
    signal x and produces a series of (index, coefficient) tuples (m_n, c_n)
    according to :

      x_1 = x
      x_{n+1} = x_n - c_n * f_n
      f_n = argmax_f <x_n, f>
      c_n = <x_n, f_n>

    This implementation of the algorithm is intended to encode signals of a
    constant shape, using filters of the same shape : 16x16 RGB image patches,
    10ms 2-channel audio clips, colors, etc.

    See the Trainer class for code that encapsulates a simple gradient ascent
    learning process for inferring codebook filters from data.
    '''

    def __init__(self, num_filters, filter_shape):
        '''Initialize a new codebook of static filters.

        num_filters: The number of filters to build in the codebook.
        filter_shape: A tuple of integers that specifies the shape of the
          filters in the codebook.
        '''
        if not isinstance(filter_shape, (tuple, list)):
            filter_shape = (filter_shape, )

        self.filters = [rng.randn(*filter_shape) for _ in range(num_filters)]
        for f in self.filters:
            f /= numpy.linalg.norm(f)

    def encode(self, signal, min_coeff=0., max_num_coeffs=-1):
        '''Encode a signal as a sequence of index, coefficient pairs.

        signal: A numpy array containing a signal to encode. The values in the
          array will be modified.
        min_coeff: Stop encoding when the maximal filter response drops below
          this threshold.
        max_num_coeffs: Stop encoding when this many filters have been used in
          the encoding.

        Generates a sequence of (index, coefficient) tuples.
        '''
        scores = numpy.array([(signal * f).sum() for f in self.filters])

        while max_num_coeffs != 0:
            max_num_coeffs -= 1

            index = scores.argmax()
            coeff = scores[index]
            if coeff < min_coeff:
                logging.debug('halting: coefficient %d is %.2f < %.2f',
                              -max_num_coeffs, coeff, min_coeff)
                break

            signal -= coeff * self.filters[index]

            scores[index] = -numpy.inf
            mask = numpy.isfinite(scores)
            scores[mask] = [(signal * f).sum()
                            for i, f in enumerate(self.filters)
                            if mask[i]]

            yield index, coeff

        else:
            logging.debug(
                'halting: final coefficient %d is %.2f', -max_num_coeffs, coeff)

    def decode(self, encoding, unused_shape):
        '''Decode an encoding of a static signal.

        encoding: A sequence of (index, coefficient) tuples.

        Returns the sum of the filters in the encoding, weighted by their
        respective coefficients.
        '''
        try:
            return sum(c * self.filters[i] for i, c in encoding)
        except TypeError:
            return numpy.zeros_like(self.filters[0])

    def encode_frames(self, frames, min_coeff=0.):
        '''Encode a sequence of frames.

        frames: A (possibly infinite) sequence of data frames to encode.
        min_coeff: Only fire for filters that exceed this threshold.

        Generates a sequence of ((index, coeff), ...) tuples at the same rate as
        the input frames. If a given input frame does not yield a filter
        coefficient better than the minimum threshold, the encoding output for
        that frame will be an empty tuple.
        '''
        # set up a circular buffer (2x the max length of a codebook vector).
        # http://mail.scipy.org/pipermail/scipy-user/2009-February/020108.html
        N = max(len(f) for f in self.filters)
        frame = self.filters[0][0]
        memory = numpy.zeros((2 * N, ) + frame.shape, frame.dtype)
        m = 0

        # make sure the buffer is fully pre-populated before encoding.
        frames = iter(frames)
        while m < N:
            memory[m] += frames.next()
            m += 1

        for frame in frames:
            # rotate the circular buffer from the back to the front if needed.
            if m == 2 * N:
                memory[:N] = memory[N:]
                memory[N:, :] = 0.
                m = N
            memory[m] += frame
            m += 1

            # calculate coefficients starting at offset m - N.
            window = memory[m - N:m]
            scores = numpy.array(
                [(window[:len(f)] * f).sum() for f in self.filters])

            encoding = []
            while True:
                index = scores.argmax()
                coeff = scores[index]
                if coeff < min_coeff:
                    logging.debug('halting: coefficient %d is %.2f < %.2f',
                                  -max_num_coeffs, coeff, min_coeff)
                    break
                encoding.append((index, coeff))
                f = self.filters[index]
                window[:len(f)] -= coeff * f
            yield tuple(encoding)

    def decode_frames(self, tuples):
        '''Given a frame encoding, decode and generate frames of output signal.

        tuples: A sequence of tuples generated by encode_frames(signal).

        Generates a sequence of signal frames at the same rate as the input
        (encoded) tuples.
        '''
        N = max(len(f) for f in self.filters)
        frame = self.filters[0][0]
        acc = numpy.zeros((2 * N, ) + frame.shape, frame.dtype)
        m = 0
        for tup in tuples:
            if m == 2 * N:
                acc[:N] = acc[N:]
                m = N
            yield acc[m]
            m += 1
            if m < N or not tup:
                continue
            index, coeff = tup
            f = self.filters[index]
            acc[m - len(f):m] += coeff * f


class Trainer(object):
    '''Train the codebook filters in a matching pursuit encoder.'''

    def __init__(self, codebook, min_coeff=0., max_num_coeffs=-1, samples=1):
        '''Initialize this trainer with some learning parameters.

        codebook: The matching pursuit codebook to train.
        min_coeff: Train by encoding signals to this minimum coefficient
          value.
        max_num_coeffs: Train by encoding signals using this many coefficients.
        samples: The number of encoding samples to draw when approximating the
          gradient.
        '''
        self.codebook = codebook
        self.min_coeff = min_coeff
        self.max_num_coeffs = max_num_coeffs
        self.samples = samples

    def calculate_gradient(self, signal):
        '''Calculate a gradient from a signal.

        signal: A signal to use for collecting gradient information. This signal
          will be modified in the course of the gradient collection process.

        Returns a pair of (gradient, activity), where activity is the sum of the
        coefficients for each codebook filter.
        '''
        grad = [numpy.zeros_like(f) for f in self.codebook.filters]
        activity = numpy.zeros((len(grad), ), float)
        for _ in range(self.samples):
            s = signal.copy()
            enc = self.codebook.encode(s, self.min_coeff, self.max_num_coeffs)
            for index, coeff, error in self._calculate_gradient(s, enc):
                grad[index] += coeff * error
                activity[index] += coeff
        return grad, activity

    def _calculate_gradient(self, error, encoding):
        '''Calculate the gradient from one encoding of a signal.'''
        for index, coeff in encoding:
            yield index, coeff, error

    def apply_gradient(self, grad, learning_rate):
        '''Apply gradients to the codebook filters.

        grad: A sequence of gradients to apply to the codebook filters.
        learning_rate: Move the codebook filters this much toward the gradients.
        '''
        for i, g in enumerate(grad):
            logging.debug('filter %d: |gradient| %.2f', i, numpy.linalg.norm(g))
            f = self.codebook.filters[i]
            f += learning_rate * g
            f /= numpy.linalg.norm(f)

    def resample(self, activity, min_activity_ratio=0.1):
        '''Create new filters to replace "inactive" ones.

        For a given codebook filter, if the activity for that  filter is below
        a specific threshold, we replace it with a new noise filter to encourage
        future usage.

        activity: A sequence of scalars indicating how active each codebook
          filter was during gradient calculation.
        min_activity_ratio: Replace filters with total activity below this
          proportion of the median activity for the codebook. This should be a
          number in [0, 1], where 0 means never replace any filters, and 1 means
          replace all filters whose activity is below the median.
        '''
        if not 0 < min_activity_ratio < 1:
            return

        median = numpy.sort(activity)[len(activity) // 2]
        logging.debug('median filter activity: %.1f', median)
        min_act = min_activity_ratio * median

        shapes = numpy.array([f.shape for f in self.codebook.filters])
        mu = shapes.mean(axis=0)
        sigma = shapes.std(axis=0)

        for i, act in enumerate(activity):
            logging.debug('filter %d: activity %.1f', i, act)
            if act < min_act:
                shape = rng.multivariate_normal(mu, numpy.diag(sigma))
                f = self.codebook.filters[i] = rng.randn(*shape)
                f /= numpy.linalg.norm(f)

    def learn(self, signal, learning_rate):
        '''Calculate and apply a gradient from the given signal.

        signal: A signal to use for collecting gradient information. This signal
          will not be modified.
        learning_rate: Move the codebook filters this much toward the gradients.

        Returns the result from the call to calculate_gradient().
        '''
        grad, activity = self.calculate_gradient(signal.copy())
        self.apply_gradient(
            (g / (a or 1) for g, a in zip(grad, activity)), learning_rate)
        return grad, activity

    def reconstruct(self, signal):
        '''Reconstruct the given signal using our pursuit codebook.

        signal: A signal to encode and then reconstruct. This signal will not
          be modified.

        Returns a numpy array with the same shape as the original signal,
        containing reconstructed values.
        '''
        return self.codebook.decode(
            self.codebook.encode(
                signal.copy(), self.min_coeff, self.max_num_coeffs, 0.),
            signal.shape)
