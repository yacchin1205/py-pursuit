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

import codebook


class Codebook(codebook.Codebook):
    '''Matching pursuit for convolving filters across the first dimension.

    The encoding process is recursive. Given a signal x(t) of length T that
    varies along dimension t, we calculate the inner product of each codebook
    filter f(t) with x(t - o) for all 0 < o < T - len(f), and then choose the
    filter f and offset o that result in the largest magnitude coefficient c. We
    subtract c * f(t) from x(t - o) and repeat the process with the new x(t).
    More formally,

      x_1(t) = x(t)
      f_n, o_n = argmax_{f,o} <x_n(t - o), f>
      c_n = <x_n(t - o_n), f_n>
      x_{n+1}(t) = x_n(t - o_n) - c_n * f_n

    where <a(t - o), b> denotes the inner product between a at offset o and b.
    (We use the correlation function to automate the dot product calculations at
    all offsets o.) The encoding consists of triples (f, c, o) for as many time
    steps n as desired. Reconstruction of the signal requires the codebook that
    was used at encoding time, plus a sequence of encoding triples: the
    reconstructed signal is just the weighted sum of the codebook filters at the
    appropriate offsets.

    Because we are processing signals of possibly variable length in dimension
    t, the codebook filters are allowed also to span different numbers of frames
    along dimension t. This makes the encoding more computationally complex, but
    the basic idea remains the same.

    This version of the algorithm is adopted from Smith and Lewicki (2006),
    "Efficient Auditory Coding" (Nature).
    '''

    def __init__(self, num_filters, filter_frames, frame_shape=()):
        '''Initialize a new codebook to a set of random filters.

        num_filters: The number of filters to use in our codebook.
        filter_frames: The length (in frames) of filters that we will use for
          our initial codebook.
        frame_shape: The shape of each frame of data that we will encode.
        '''
        super(Codebook, self).__init__(
            num_filters, (filter_frames, ) + frame_shape)

        self.filters = list(self.filters)

        self._correlate = codebook.default_correlate
        if codebook.have_correlate and len(frame_shape) == 0:
            self._correlate = codebook._correlate.correlate1d
        if codebook.have_correlate and len(frame_shape) == 1:
            self._correlate = codebook._correlate.correlate1d_from_2d

    def encode(self, signal, min_coeff=0., max_num_coeffs=-1):
        '''Generate a set of codebook coefficients for encoding a signal.

        signal: A signal to encode.
        min_coeff: Stop encoding when the magnitude of coefficients falls below
          this threshold. Use 0 to encode until max_num_coeffs is reached.
        max_num_coeffs: Stop encoding when we have generated this many
          coefficients. Use a negative value to encode until min_coeff is
          reached.

        This method generates a sequence of tuples of the form (index,
        coefficient, offset), where index refers to a codebook filter and
        coefficient is the scalar multiple of the filter that is present in the
        input signal starting at the given offset.

        See the TemporalTrainer class for an example of how to use these results
        to update the codebook filters.
        '''
        width = signal.shape[0]
        shapes = [f.shape[0] for f in self.filters]

        # we cache the correlations between signal and codebook to avoid
        # redundant computation.
        shape = (len(self.filters), width - min(shapes) + 1)
        scores = numpy.zeros(shape, float) - numpy.inf
        for i, f in enumerate(self.filters):
            self._correlate(signal, f, scores[i, :width - shapes[i] + 1])

        amplitude = abs(signal).sum()
        while max_num_coeffs != 0:
            max_num_coeffs -= 1

            # find the largest coefficient, check that it's large enough.
            index, offset = numpy.unravel_index(scores.argmax(), shape)
            end = offset + shapes[index]
            coeff = scores[index, offset]
            if coeff < min_coeff:
                logging.debug('halting: coefficient %d is %.2f < %.2f',
                              -max_num_coeffs, coeff, min_coeff)
                break

            # check that using this filter does not increase signal amplitude by
            # more than 1 %.
            a = amplitude - abs(signal[offset:end]).sum()
            signal[offset:end] -= coeff * self.filters[index]
            a += abs(signal[offset:end]).sum()
            #logging.debug('coefficient %.2f, filter %d, offset %d yields amplitude %.2f', coeff, index, offset, a)
            if a > 1.01 * amplitude:
                logging.debug('halting: coefficient %.2f, filter %d, '
                              'offset %d yields amplitude %.2f > 1.01 * %.2f',
                              coeff, index, offset, a, amplitude)
                break
            amplitude = a

            # update the correlation cache for the changed part of signal.
            for i, f in enumerate(self.filters):
                l = shapes[i] - 1
                o = max(0, offset - l)
                p = min(end, len(signal) - l)
                self._correlate(signal[o:end + l], f, scores[i, o:p])

            yield index, coeff, offset

        else:
            logging.debug(
                'halting: final coefficient %d is %.2f', -max_num_coeffs, coeff)

    def decode(self, coefficients, signal_shape):
        '''Decode a dictionary of codebook coefficients as a signal.

        coefficients: A sequence of (index, coefficient, offset) tuples.
        signal_shape: The shape of the reconstructed signal.

        Returns a signal that consists of the weighted sum of the codebook
        filters given in the encoding coefficients, at the appropriate offsets.
        '''
        signal = numpy.zeros(signal_shape, float)
        for index, coeff, offset in coefficients:
            f = self.filters[index]
            signal[offset:offset + len(f)] += coeff * f
        return signal


class Trainer(codebook.Trainer):
    '''Train a set of temporal codebook filters using signal data.'''

    def _calculate_gradient(self, error, encoding):
        '''Calculate the gradient from one encoding of a signal.'''
        for index, coeff, offset in encoding:
            f = self.codebook.filters[index]
            yield index, coeff, error[offset:offset + len(f)]

    def _resize(self, i, padding, shrink, grow):
        '''Resize codebook vector i using some energy heuristics.

        i: The index of the codebook vector to resize.
        padding: The proportion of each codebook filter to consider as "padding"
          when growing or shrinking. Values around 0.1 are usually good. 0
          disables growing or shrinking of the codebook filters.
        shrink: Remove the padding from a codebook filter when the signal in the
          padding falls below this threshold.
        grow: Add padding to a codebook filter when signal in the padding
          exceeds this threshold.
        '''
        f = self.codebook.filters[i]
        p = int(numpy.ceil(len(f) * padding))
        criterion = abs(numpy.concatenate([f[:p], f[-p:]])).mean()
        logging.debug('filter %d: resize criterion %.3f', i, criterion)
        if criterion < shrink and len(f) > 1 + 2 * p:
            return f[p:-p]
        if criterion > grow:
            pad = numpy.zeros((p, ) + f.shape[1:], f.dtype)
            return numpy.concatenate([pad, f, pad])
        return f
