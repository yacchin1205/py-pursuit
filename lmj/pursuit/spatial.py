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
        super(SpatialCodebook, self).__init__(
            num_filters, filter_shape + ((channels, ) if channels else ()))

        self.filters = list(self.filters)

        self._correlate = codebook.default_correlate
        if codebook.have_correlate and channels == 0:
            self._correlate = codebook._correlate.correlate2d
        if codebook.have_correlate and channels == 3:
            self._correlate = codebook._correlate.correlate2d_from_rgb

    def encode(self, signal, min_coeff=0., max_num_coeffs=-1, noise=0.):
        '''Generate a set of codebook coefficients for encoding a signal.

        signal: A signal to encode.
        min_coeff: Stop encoding when the magnitude of coefficients falls below
          this threshold. Use 0 to encode until max_num_coeffs is reached.
        max_num_coeffs: Stop encoding when we have generated this many
          coefficients. Use a negative value to encode until min_coeff is
          reached.
        noise: Coefficients are chosen based on their responses to the signal,
          plus white noise with this standard deviation. Use 0 to get the
          traditional argmax behavior.

        This method generates a sequence of tuples of the form (index,
        coefficient, (x offset, y offset)), where index refers to a codebook
        filter and coefficient is the scalar multiple of the filter that is
        present in the input signal starting at the given offsets.

        See the SpatialTrainer class for an example of how to use these
        results to update the codebook filters.
        '''
        width, height = signal.shape[:2]
        shapes = [f.shape[:2] for f in self.filters]

        # we cache the correlations between signal and codebook to avoid
        # redundant computation.
        shape = (len(self.filters),
                 width - min(w for w, _ in shapes) + 1,
                 height - min(h for _, h in shapes) + 1)
        scores = numpy.zeros(shape, float) - numpy.inf
        for i, f in enumerate(self.filters):
            w, h = shapes[i]
            self._correlate(signal, f, scores[i, :width-w+1, :height-h+1])

        blur = noise * rng.randn(*scores.shape)

        amplitude = abs(signal).sum()
        while max_num_coeffs != 0:
            max_num_coeffs -= 1

            if noise > 0 and 0 == max_num_coeffs % 10:
                blur = noise * rng.randn(*scores.shape)

            # find the largest coefficient, check that it's large enough.
            index, x, y = numpy.unravel_index(
                (scores + blur).argmax(), shape)
            ex = x + shapes[index][0]
            ey = y + shapes[index][1]
            coeff = scores[index, x, y]
            if coeff < min_coeff:
                logging.debug('halting: coefficient %d is %.2f < %.2f',
                              -max_num_coeffs, coeff, min_coeff)
                break

            # check that using this filter does not increase signal amplitude by
            # more than 1 %.
            a = amplitude - abs(signal[x:ex, y:ey]).sum()
            signal[x:ex, y:ey] -= coeff * self.filters[index]
            a += abs(signal[x:ex, y:ey]).sum()
            #logging.debug('coefficient %.2f, filter %d, offset %s yields amplitude %.2f', coeff, index, (x, y), a)
            if a > 1.01 * amplitude:
                logging.debug('halting: coefficient %.2f, filter %d, '
                              'offset %s yields amplitude %.2f > 1.01 * %.2f',
                              coeff, index, (x, y), a, amplitude)
                break
            amplitude = a

            # update the correlation cache for the changed part of signal.
            for i, f in enumerate(self.filters):
                wx, wy = shapes[i][0] - 1, shapes[i][1] - 1
                ox, oy = max(0, x - wx), max(0, y - wy)
                px, py = min(ex, width - wx), min(ey, height - wy)
                self._correlate(
                    signal[ox:ex + wx, oy:ey + wy], f, scores[i, ox:px, oy:py])

            yield index, coeff, (x, y)

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
        for index, coeff, (x, y) in coefficients:
            f = self.filters[index]
            w, h = f.shape[:2]
            signal[x:x + w, y:y + h] += coeff * f
        return signal


class Trainer(codebook.Trainer):
    '''Train a set of spatial codebook filters using signal data.'''

    def _calculate_gradient(self, error, encoding):
        '''Calculate the gradient from one encoding of a signal.'''
        for index, coeff, (x, y) in encoding:
            w, h = self.codebook.filters[index].shape[:2]
            yield index, coeff, error[x:x + w, y:y + h]

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
