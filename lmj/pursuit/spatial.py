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
from itertools import izip as zip

import codebook


class Codebook(codebook.Codebook):
    '''A matching pursuit for encoding images or other 2D signals.'''

    def __init__(self, num_filters, filter_shape, channels=0):
        '''Initialize a new codebook of static filters.

        num_filters: The number of filters to build in the codebook.
        filter_shape: A tuple of integers that specifies the shape of the
          filters in the codebook.
        channels: Set this to the number of channels in each element of the
          signal (and the filters). Leave this set to 0 if your 2D signals
          have just two values in their shape tuples.
        '''
        super(Codebook, self).__init__(
            num_filters, filter_shape + ((channels, ) if channels else ()))

        self.filters = list(self.filters)

        self._correlate = codebook.default_correlate
        if codebook.have_correlate and channels == 0:
            self._correlate = codebook._correlate.correlate2d
        if codebook.have_correlate and channels == 3:
            self._correlate = codebook._correlate.correlate2d_from_rgb

    def encode(self, signal, min_coeff=0., max_num_coeffs=-1):
        '''Generate a set of codebook coefficients for encoding a signal.

        signal: A signal to encode.
        min_coeff: Stop encoding when the magnitude of coefficients falls below
          this threshold. Use 0 to encode until max_num_coeffs is reached.
        max_num_coeffs: Stop encoding when we have generated this many
          coefficients. Use a negative value to encode until min_coeff is
          reached.

        This method generates a sequence of tuples of the form (index,
        coefficient, (x offset, y offset)), where index refers to a codebook
        filter and coefficient is the scalar multiple of the filter that is
        present in the input signal starting at the given offsets.

        See the SpatialTrainer class for an example of how to use these
        results to update the codebook filters.
        '''
        signal_shape = numpy.array(signal.shape)
        filter_shapes = numpy.array([f.shape for f in self.filters])

        # we cache the correlations between signal and codebook to avoid
        # redundant computation.
        shape = tuple(signal_shape - shapes.min(axis=0) + 1)
        scores = numpy.zeros((len(self.filters), ) + shape, float) - numpy.inf
        for i, f in enumerate(self.filters):
            target = (slice(0, x) for x in signal_shape - filter_shapes[i] + 1)
            self._correlate(signal, f, scores[(i, ) + tuple(target)])

        amplitude = abs(signal).sum()
        while max_num_coeffs != 0:
            max_num_coeffs -= 1

            # find the largest coefficient, check that it's large enough.
            whence = numpy.unravel_index(scores.argmax(), scores.shape)
            coeff = scores[whence]
            if coeff < min_coeff:
                logging.debug('halting: coefficient %d is %.2f < %.2f',
                              -max_num_coeffs, coeff, min_coeff)
                break

            index, offset = whence[0], whence[1:]
            region = tuple(
                slice(o, o + s) for o, s in zip(offset, filter_shapes[index]))

            # check that using this filter does not increase signal amplitude by
            # more than 1 %.
            a = amplitude - abs(signal[region]).sum()
            signal[region] -= coeff * self.filters[index]
            a += abs(signal[region]).sum()
            #logging.debug('coefficient %.2f, filter %d, offset %r yields amplitude %.2f', coeff, index, offset, a)
            if a > 1.01 * amplitude:
                logging.debug('halting: coefficient %.2f, filter %d, '
                              'offset %s yields amplitude %.2f > 1.01 * %.2f',
                              coeff, index, offset, a, amplitude)
                break
            amplitude = a

            # update the correlation cache for the changed part of signal.
            for i, f in enumerate(self.filters):
                o = max(0, offset - filter_shape[i] + 1)
                p = min(offset + filter_shapes[i], signal_shape - filter_shapes[i] + 1)
                source = tuple(slice(max(0, o - fs + 1), o + fs + fs - 1)
                               for o, fs in zip(offset, filter_shape[i]))
                target = tuple(
                    slice(max(0, o - fs + 1), min(o + fs, ss - fs + 1))
                    for o, fs, ss in zip(offset, shape, signal_shape))
                self._correlate(signal[source], f, scores[(i, ) + target])

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
            region = tuple(slice(o, o + s) for o, s in zip(offset, f.shape))
            signal[region] += coeff * f
        return signal


class Trainer(codebook.Trainer):
    '''Train a set of spatial codebook filters using signal data.'''

    def _calculate_gradient(self, error, encoding):
        '''Calculate the gradient from one encoding of a signal.'''
        for index, coeff, offset in encoding:
            f = self.codebook.filters[index]
            region = tuple(slice(o, o + s) for o, s in zip(offset, f.shape))
            yield index, coeff, error[region]

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
        w, h = f.shape[:2]
        p = int(numpy.ceil(w * padding))
        q = int(numpy.ceil(h * padding))
        criterion = abs(numpy.concatenate([
            f[:p].flatten(),
            f[-p:].flatten(),
            f[p:-p, :q].flatten(),
            f[p:-p, -q:].flatten()])).mean()
        logging.debug('filter %d: resize criterion %.3f', i, criterion)
        if criterion < shrink and w > 1 + 2 * p and h > 1 + 2 * q:
            return f[p:-p, q:-q]
        if criterion > grow:
            ppad = numpy.zeros((p, h) + f.shape[2:], f.dtype)
            qpad = numpy.zeros((2 * p + w, q) + f.shape[2:], f.dtype)
            return numpy.hstack(
                [qpad, numpy.concatenate([ppad, f, ppad]), qpad])
        return f
