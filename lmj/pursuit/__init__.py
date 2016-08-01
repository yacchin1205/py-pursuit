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
recursively decomposing the residual. Encoding stops after either a fixed
number of filters have been used, or until the maximal filter response drops
below some threshold. The encoding thus represents a signal as a weighted sum of
filters, with many of the weights being zero.

This module contains two implementations of matching pursuit: one for encoding
signals of a fixed shape using filters of the same shape (the Codebook class),
and another for encoding signals and filters of the same number of dimensions,
using correlations across dimensions (the correlation.Codebook class). Each
implementation comes with an associated Trainer class that encapsulates the
logic involved with basic gradient-ascent training for the filters.

An experimental CUDA implementation of the correlation approach is also
included ; to use it, "import lmj.pursuit.cuda" and then use the Codebook class
in that module.
'''

# Reference implementation of the greedy Mallat/Zhang algorithm.
from codebook import Codebook, Trainer

# Reference implementation of a matching pursuit based on correlating an input
# signal with codebook filters.
from correlation import Codebook as CorrelationCodebook
from correlation import Trainer as CorrelationTrainer
