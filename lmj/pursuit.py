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

'''Matching pursuit sparse coding algorithm and gradient ascent trainer.

Matching pursuit is a greedy sparse coding algorithm originally presented by
Mallat and Zhang (1993, IEEE Trans Sig Proc), "Matching Pursuits with
Time-Frequency Dictionaries." Using a fixed codebook (bank, etc.) of filters
(basis functions, signals, vectors, etc.), the algorithm decomposes a signal
(function, vector, etc.) into the maximally responding filter and a residual,
recursively decomposing the residual using the remaining filters in the
codebook. Encoding stops after either a fixed number of filters have been used,
or until the maximal filter response drops below some threshold. The encoding
thus represents a signal as a weighted sum of the filters, with many of the
weights being zero.

This module contains two implementations of matching pursuit: one for encoding
signals consisting of a single frame of data (the Single class), and another for
encoding signals consisting of multiple frames of data (the Multiple class).
Each implementation also comes with a Trainer class that encapsulates the logic
involved with basic gradient-ascent training for the codebook filters.
'''

import numpy
import logging


class Single(object):
    '''This class encodes signals consisting of a single frame.

    The encoding process decomposes a signal recursively into a maximally
    responding filter and a residual. Formally, the encoding process takes a
    signal s and produces a series of (index, coefficient) tuples (m_n, c_n)
    according to :

      s_1 = s
      w_n = argmax_w <s_n, w>
      c_n = <s_n, w_n>
      s_{n+1} = s_n - c_n * w_n

    This implementation of the algorithm is intended to encode signals of a
    constant shape : 16x16 RGB image patches, 10ms 2-channel audio clips,
    colors, etc.

    See the SingleTrainer class for code that encapsulates a simple gradient
    ascent learning process for inferring codebook filters from data.
    '''

    def __init__(self, num_filters, filter_shape, dtype=float):
        '''Initialize a new codebook of static filters.

        num_filters: The number of filters to build in the codebook.
        filter_shape: A tuple of integers that specifies the shape of the
          filters in the codebook.
        dtype: The numpy data type of the filters.
        '''
        self.dtype = dtype
        if not isinstance(filter_shape, (tuple, list)):
            filter_shape = (filter_shape, )
        self.codebook = numpy.random.randn(
            num_filters, *filter_shape).astype(dtype)
        for w in self.codebook:
            w /= numpy.linalg.norm(w)

    def iterencode(self, signal, min_coeff=0., max_num_coeffs=-1):
        '''Encode a signal as a sequence of index, coefficient pairs.

        signal: A numpy array containing a signal to encode. The values in the
          array will be modified.
        min_coeff: Stop encoding when the maximal filter response drops below
          this threshold.
        max_num_coeffs: Stop encoding when this many filters have been used in
          the encoding.

        Generates a sequence of (index, coefficient) tuples.
        '''
        coeffs = numpy.array([(signal * w).sum() for w in self.codebook])
        while max_num_coeffs != 0:
            max_num_coeffs -= 1

            index = coeffs.argmax()
            coeff = coeffs[index]
            if coeff < min_coeff:
                break

            signal -= coeff * self.codebook[index]

            coeffs[index] = -numpy.inf
            mask = numpy.isfinite(coeffs)
            coeffs[mask] = [(signal * w).sum() for w in self.codebook[mask]]

            yield index, coeff

    def encode(self, signal, min_coeff=0., max_num_coeffs=-1):
        '''Encode a signal using our dictionary of codebook filters.

        signal: A numpy array to encode. This signal must be the same shape as
          the filters in our codebook. The values in the signal array will be
          modified.
        min_coeff: Stop encoding when the maximal filter response drops below
          this threshold.
        max_num_coeffs: Stop encoding when this many filters have been used in
          the encoding.

        Returns a tuple of (index, coefficient) tuples.
        '''
        return tuple(self.iterencode(signal, min_coeff, max_num_coeffs))

    def decode(self, encoding):
        '''Decode an encoding of a static signal.

        encoding: A sequence of (index, coefficient) tuples.

        Returns the sum of the filters in the encoding, weighted by their
        respective coefficients.
        '''
        try:
            return sum(c * self.codebook[i] for i, c in encoding)
        except:
            return numpy.zeros_like(self.codebook[0])


class SingleTrainer(object):
    '''Train the codebook filters in a single-frame matching pursuit encoder.
    '''

    def __init__(self, pursuit, min_coeff=0., max_num_coeffs=-1, momentum=0., l1=0., l2=0.):
        '''Initialize this trainer with some learning parameters.

        pursuit: The matching pursuit object to train.
        min_coeff: Train by encoding signals to this minimum coefficient
          value.
        max_num_coeffs: Train by encoding signals using this many coefficients.
        momentum: Use this momentum value during gradient descent.
        l1: L1-regularize the codebook filters with this weight.
        l2: L2-regularize the codebook filters with this weight.
        '''
        self.pursuit = pursuit

        self.min_coeff = min_coeff
        self.max_num_coeffs = max_num_coeffs

        self.momentum = momentum
        self.l1 = l1
        self.l2 = l2

        self.grad = numpy.zeros_like(self.pursuit.codebook)

    def calculate_gradient(self, signal):
        '''Calculate a gradient from a signal.

        signal: A signal to use for collecting gradient information. This signal
          will be modified in the course of the gradient collection process.
        '''
        grad = numpy.zeros_like(self.grad)
        norm = [0] * len(grad)
        for index, coeff in self.pursuit.iterencode(
                signal, self.min_coeff, self.max_num_coeffs):
            grad[index] += coeff * signal
            norm[index] += coeff
        return (g / (n or 1) for g, n in zip(grad, norm))

    def apply_gradient(self, grad, learning_rate):
        '''Apply gradients to the codebook filters.

        grad: A sequence of gradients to apply to the codebook filters.
        learning_rate: Move the codebook filters this much toward the gradients.
        '''
        for w, g, sg in zip(self.pursuit.codebook, grad, self.grad):
            l1 = numpy.clip(w, -self.l1, self.l1)
            l2 = self.l2 * w
            sg *= self.momentum
            sg += (1 - self.momentum) * (g - l1 - l2)
            w += learning_rate * sg

    def learn(self, signal, learning_rate):
        '''Calculate and apply a gradient from the given signal.

        signal: A signal to use for collecting gradient information. This signal
          will not be modified.
        learning_rate: Move the codebook filters this much toward the gradients.
        '''
        self.apply_gradient(
            self.calculate_gradient(signal.copy()), learning_rate)

    def reconstruct(self, signal):
        '''Reconstruct the given signal using our pursuit codebook.

        signal: A signal to encode and then reconstruct. This signal will not
          be modified.

        Returns a numpy array with the same shape as the original signal,
        containing reconstructed values instead of the original values.
        '''
        return self.pursuit.decode(self.pursuit.iterencode(
            signal.copy(), self.min_coeff, self.max_num_coeffs))


class Multiple(object):
    '''Matching pursuit for multiple-frame signals.

    The encoding process is recursive. Given a signal s, we calculate the inner
    product of each codebook filter w with s, and then choose the filter w and
    offset t that result in the largest magnitude coefficient c. We subtract
    c * w from s[t:t+len(w)] and repeat the process with the new s. More
    formally,

      s_1 = s
      w_n, t_n = argmax_{w,t} <s_n[t], w>
      c_n = <s_n[t_n], w_n>
      s_{n+1} = s_n[t_n] - c_n * w_n

    where <a[t], b> denotes the dot product between a (starting at offset t) and
    b. (We use the correlation function to automate the dot product calculations
    at all offsets t.) The encoding consists of triples (w, c, t) for as many
    time steps n as desired. Reconstruction of the signal requires the codebook
    that was used at encoding time, plus a sequence of encoding triples: the
    reconstructed signal is just the weighted sum of the codebook filters at the
    appropriate offsets.

    Because we are processing sequences of frames of possibly variable length,
    the codebook filters are allowed also to span different numbers of frames.
    This makes the encoding more computationally complex, but the basic idea
    remains the same.

    This version of the algorithm is inspired by Smith and Lewicki (2006),
    "Efficient Auditory Coding" (Nature).
    '''

    def __init__(self, num_filters, filter_frames, frame_shape=(), dtype=float):
        '''Initialize a new codebook to a set of random filters.

        num_filters: The number of filters to use in our codebook.
        filter_frames: The length (in frames) of filters that we will use for
          our encoding.
        frame_shape: The shape of each frame of data that we will encode.
        dtype: The data type of our filters.
        '''
        self.dtype = dtype
        self.codebook = [
            numpy.random.randn(filter_frames, *frame_shape).astype(dtype)
            for _ in range(num_filters)]
        for w in self.codebook:
            w /= numpy.linalg.norm(w)

        f = len(frame_shape)
        if f < 2:
            import _pursuit
            self._correlate = getattr(_pursuit, 'correlate%dd' % (f + 1))
        else:
            import scipy.signal
            def correlate(s, w, r):
                r[:] = scipy.signal.correlate(s, w, 'valid')
            self._correlate = correlate

    def iterencode(self, signal, min_coeff=0., max_num_coeffs=-1):
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

        See the MultipleTrainer class for an example of how to use these results
        to update the codebook filters.
        '''
        def rmsp(s):
            return numpy.linalg.norm(s) / numpy.sqrt(len(s))

        lengths = [len(w) for w in self.codebook]
        arange = numpy.arange(len(self.codebook))

        # we cache the correlations between signal and codebook to avoid
        # redundant computation.
        scores = numpy.zeros(
            (len(self.codebook), len(signal) - min(lengths) + 1),
            float) - numpy.inf
        for i, w in enumerate(self.codebook):
            self._correlate(signal, w, scores[i, :len(signal) - len(w) + 1])

        power = rmsp(signal)
        while max_num_coeffs != 0:
            max_num_coeffs -= 1

            # find the largest coefficient, check that it's large enough.
            offsets = scores.argmax(axis=1)
            coeffs = scores[arange, offsets]
            index = coeffs.argmax()
            coeff = coeffs[index]
            offset = offsets[index]
            length = lengths[index]
            end = offset + length
            if coeff < min_coeff:
                break

            # check that using this filter does increase signal power.
            signal[offset:end] -= coeff * self.codebook[index]
            r = rmsp(signal)
            #logging.debug('coefficient %.3g, filter %d, offset %d yields power %.3g', coeff, index, offset, r)
            if r > power:
                break
            power = r

            # update the correlation cache for the changed part of signal.
            for i, w in enumerate(self.codebook):
                l = lengths[i] - 1
                o = max(0, offset - l)
                self._correlate(signal[o:end + l], w, scores[i, o:min(end, len(signal) - l)])

            yield index, coeff, offset

    def encode(self, signal, min_coeff=0., max_num_coeffs=-1):
        '''Encode a signal as a set of filter coefficients.

        signal: A signal to encode.
        min_coeff: Stop encoding when the magnitude of coefficients falls below
          this threshold. Use 0 to encode until max_num_coeffs is reached.
        max_num_coeffs: Stop encoding when we get this many coefficients. Use a
          negative value to encode until min_coeff is reached.

        Returns a tuple of encoding tuples.
        '''
        return tuple(self.iterencode(signal, min_coeff, max_num_coeffs))

    def decode(self, coefficients, signal_shape):
        '''Decode a dictionary of codebook coefficients as a signal.

        coefficients: A sequence of (index, coefficient, offset) tuples.
        signal_shape: The shape of the reconstructed signal.

        Returns a signal that consists of the weighted sum of the codebook
        filters given in the encoding coefficients, at the appropriate offsets.
        '''
        signal = numpy.zeros(signal_shape, self.dtype)
        for index, coeff, offset in coefficients:
            w = self.codebook[index]
            signal[offset:offset + len(w)] += coeff * w
        return signal


class MultipleTrainer(object):
    '''Train a set of codebook filters using signal data.'''

    def __init__(self, pursuit,
                 min_coeff=0., max_num_coeffs=-1,
                 momentum=0., l1=0., l2=0.,
                 padding=0.1, shrink=0.001, grow=0.1):
        '''Set up the trainer with some static learning parameters.

        pursuit: The matching pursuit object to train.
        min_coeff: Train by encoding signals to this minimum coefficient
          value.
        max_num_coeffs: Train by encoding signals using this many coefficients.
        momentum: Use this momentum value during gradient descent.
        l1: L1-regularize the codebook filters with this weight.
        l2: L2-regularize the codebook filters with this weight.
        padding: The proportion of each codebook filter to consider as "padding"
          when growing or shrinking. Values around 0.1 are usually good. None
          disables growing or shrinking of the codebook filters.
        shrink: Remove the padding from a codebook filter when the signal in the
          padding falls below this threshold.
        grow: Add padding to a codebook filter when signal in the padding
          exceeds this threshold.
        '''
        self.pursuit = pursuit

        self.min_coeff = min_coeff
        self.max_num_coeffs = max_num_coeffs

        self.momentum = momentum
        self.l1 = l1
        self.l2 = l2

        self.padding = padding
        self.shrink = shrink
        self.grow = grow

        self.grad = [numpy.zeros_like(w) for w in self.pursuit.codebook]

    def calculate_gradient(self, signal):
        '''Calculate a gradient from a signal.

        signal: A signal to use for collecting gradient information. This signal
          will be modified in the course of the gradient collection process.
        '''
        grad = [numpy.zeros_like(g) for g in self.grad]
        norm = [0.] * len(grad)
        for index, coeff, offset in self.pursuit.iterencode(
                signal, self.min_coeff, self.max_num_coeffs):
            o = len(self.pursuit.codebook[index])
            grad[index] += coeff * signal[offset:offset + o]
            norm[index] += coeff
        return (g / (n or 1) for g, n in zip(grad, norm))

    def apply_gradient(self, grad, learning_rate):
        '''Apply gradients to the codebook filters.

        grad: A list of gradients to apply to the codebook filters.
        learning_rate: Move the codebook filters this much toward the gradients.
        '''
        for i, g in enumerate(grad):
            w = self.pursuit.codebook[i]

            l1 = numpy.clip(w, -self.l1, self.l1)
            l2 = self.l2 * w
            self.grad[i] *= self.momentum
            self.grad[i] += (1 - self.momentum) * (g - l1 - l2)
            w += learning_rate * self.grad[i]

            # expand or clip the codebook filter based on activity.
            p = int(len(w) * self.padding)
            criterion = numpy.concatenate([abs(w)[:p], abs(w)[-p:]]).mean()
            #logging.debug('filter criterion %.3g', criterion)
            if criterion < self.shrink:
                w = self.pursuit.codebook[i] = w[p:-p]
                self.grad[i] = self.grad[i][p:-p]
            elif criterion > self.grow:
                pad = numpy.zeros((p, ) + w.shape[1:], w.dtype)
                w = self.pursuit.codebook[i] = numpy.concatenate([pad, w, pad])
                self.grad[i] = numpy.concatenate([pad, self.grad[i], pad])

    def learn(self, signal, learning_rate):
        '''Calculate and apply a gradient from the given signal.

        signal: A signal to use for collecting and applying gradient data.
          This signal will not be modified.
        learning_rate: Move the codebook filters this much toward the gradients.
        '''
        self.apply_gradient(
            self.calculate_gradient(signal.copy()), learning_rate)

    def reconstruct(self, signal):
        '''Reconstruct the given signal using our pursuit codebook.

        signal: A signal to encode and then reconstruct. This signal will not
          be modified.

        Returns a numpy array with the same shape as the original signal,
        containing reconstructed values instead of the original values.
        '''
        return self.pursuit.decode(self.pursuit.iterencode(
            signal.copy(), self.min_coeff, self.max_num_coeffs), signal.shape)
