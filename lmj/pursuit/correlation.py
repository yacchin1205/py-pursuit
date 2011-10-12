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

_correlate = None
try:
    import _correlate
except ImportError:
    logging.info('cannot import _correlate module, trying scipy')
    import scipy.signal

def _default_correlate(s, f, r):
    '''Assign to r the values from scipy.signal.correlate(s, f).'''
    r[:] = scipy.signal.correlate(s, f, 'valid')


class Codebook(codebook.Codebook):
    '''A matching pursuit for encoding signals using pointwise correlation.

    The basic matching pursuit encodes signals that are the same size as the
    filters in the codebook. This implementation generalizes that process so
    that filters and signals must only share the same number of dimensions.
    Additionally, signals must be the same size or larger than all codebook
    filters in all dimensions, but this is the normal case -- we usually encode
    100x100 images using small 8x8 or 16x16, etc. filter patches, for instance.

    In addition to generating the codebook filter index, and the corresponding
    coefficient, this implementation correlates each filter across the entire
    signal, generating the maximally responding codebook filter, coefficient,
    and offset. The decoded signal is then a weighted reconstruction of filters
    offset by the appropriate number of samples in each dimension.

    Additionally, because codebook filters are correlated across the signals, we
    can resize codebook vectors if necessary to capture more information in the
    dataset being learned. See the Trainer.resize method for details.
    '''

    def __init__(self, num_filters, filter_shape):
        '''Initialize a new codebook of filters.

        num_filters: The number of filters to build in the codebook.
        filter_shape: A tuple of integers that specifies the shape of the
          filters in the codebook.
        '''
        super(Codebook, self).__init__(num_filters, filter_shape)

        try:
            self._correlate = getattr(
                _correlate, 'correlate%dd' % len(self.filters[0].shape))
        except Exception, e:
            logging.exception('error')
            self._correlate = _default_correlate

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
        input signal starting at the given offsets.

        See the Trainer class for an example of how to use these results to
        update the codebook filters.
        '''
        signal_shape = numpy.array(signal.shape)
        filter_shapes = numpy.array([f.shape for f in self.filters])

        # we cache the correlations between signal and codebook to avoid
        # redundant computation.
        shape = tuple(signal_shape - filter_shapes.min(axis=0) + 1)
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
            #
            # we just changed the signal in the "affected region" from offset to
            # offset + filter_shape[index]. so we need to update the scores for
            # that region. to do that, we need that corresponding part of the
            # signal, plus all the "fringe" areas that the filters will overlap
            # during the correlation computation.
            #
            # so, we wrap up the offset, filter shape, and signal shape. for
            # each dimension, we start both the scores and the signal region at
            # offset - filter_shape + 1, clipped to 0. (in my code, the
            # correlation always shares the same starting point, while the end
            # absorbs all of the fringe.)
            #
            # we end the source region for the signal at (offset + filter_shape)
            # + filter_shape - 1, basically the end of the affected region
            # plus the overlap due to the filter, clipped to the signal_shape.
            #
            # we end the target region for the scores at the end of the affected
            # region (offset + filter_shape), clipped to the end of the valid
            # signal region (signal_shape - filter_shape + 1).
            for i, f in enumerate(self.filters):
                source = []
                target = [i]
                for o, fs, ss in zip(offset, filter_shapes[i], signal_shape):
                    a = max(0, o - fs + 1)
                    source.append(slice(a, min(ss, o + fs + fs - 1)))
                    target.append(slice(a, min(o + fs, ss - fs + 1)))
                self._correlate(signal[tuple(source)], f, scores[tuple(target)])

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

    def resize(self, paddings, shrink, grow):
        '''Resize the filters in our codebook.

        paddings: For each axis in our filters, this provides a floating point
          proportion of each codebook filter to consider as "padding"
          when growing or shrinking along this axis. Typically 0.1 or so. Values
          of 0 in the sequence disable growing or shrinking of the codebook
          filters along the associated axis.
        shrink: Remove the padding from a codebook filter when the signal in the
          padding falls below this threshold.
        grow: Add padding to a codebook filter when signal in the padding
          exceeds this threshold.
        '''
        fs = self.codebook.filters
        if isinstance(paddings, (int, float)):
            paddings = [paddings] * fs[0].ndim
        assert len(paddings) == fs[0].ndim
        for i, f in enumerate(fs):
            fs[i] = self._resize(i, paddings, shrink, grow)
            fs[i] /= numpy.linalg.norm(fs[i])

    def _resize(self, i, paddings, shrink, grow):
        '''Resize codebook vector i using a signal magnitude heuristic.

        i: The index of the codebook vector to resize.
        paddings: For each axis in our filters, this sequence provides the
          proportion of each codebook filter to consider as "padding"
          when growing or shrinking.
        shrink: Remove the padding from a codebook filter when the signal in the
          padding falls below this threshold.
        grow: Add padding to a codebook filter when signal in the padding
          exceeds this threshold.
        '''
        f = self.codebook.filters[i]
        for j, p in enumerate(paddings):
            s = f.shape[j]
            p = int(numpy.ceil(s * p))
            if not p:
                continue
            w = tuple(slice(None) for _ in range(j))
            criterion = abs(numpy.append(
                f[w + (slice(p), )], f[w + (slice(-p, None), )])).mean()
            logging.debug('filter %d:%d: resize %.3f', i, j, criterion)
            if criterion < shrink and s > 1 + 2 * p:
                f = f[w + (slice(p, -p), )]
            if criterion > grow:
                pad = numpy.zeros(f.shape[:j] + (p, ) + f.shape[j+1:], f.dtype)
                f = numpy.append(numpy.append(pad, f, axis=j), pad, axis=j)
        return f
