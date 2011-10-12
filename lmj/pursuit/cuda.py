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

import sys
import time
import numpy
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
import pycuda.gpuarray

import codebook

THREADS_PER_BLOCK = 512

SOURCE = '''
#define THREADS_PER_BLOCK %d

typedef %s real;
typedef %s integer;

#define FILTER_X %d
#define FILTER_Y %d
#define FILTER_Z %d

__device__ __constant__ real FILTER[FILTER_X * FILTER_Y * FILTER_Z];

__global__ void correlate(real *result, const real *signal, const integer x, const integer y, const integer z)
{
  const integer n = x * y * z;
  const integer tid = threadIdx.x;
  const integer i = blockIdx.x * THREADS_PER_BLOCK + tid;
  __shared__ real data[(THREADS_PER_BLOCK + FILTER_X) * FILTER_Y * FILTER_Z];
  for (i > 0 && i % x == 0) {
%s
  }
  __syncthreads();
  if (i < (x - FILTER_X + 1) * (y - FILTER_Y + 1) * (z - FILTER_Z + 1)) {
    real acc = 0;
%s
    result[i] = acc;
  }
}

#define REDUCE(i) { if (vals[tid] <= vals[tid + i]) { vals[tid] = vals[tid + i]; idxs[tid] = idxs[tid + i]; } }

__global__ void argmax(integer *index, real *value, const real *data, const integer n)
{
  const integer tid = threadIdx.x;
  __shared__ real vals[THREADS_PER_BLOCK];
  __shared__ integer idxs[THREADS_PER_BLOCK];
  real val = (tid < n) ? data[tid] : data[0] + 1;
  integer idx = (tid < n) ? tid : -1;
  for (integer i = tid; i < n; i += THREADS_PER_BLOCK)
    if (val <= data[i]) { val = data[i]; idx = i; }
  vals[tid] = val; idxs[tid] = idx; __syncthreads();
#if (THREADS_PER_BLOCK >= 512)
  if (tid < 256) REDUCE(256) __syncthreads();
#endif
#if (THREADS_PER_BLOCK >= 256)
  if (tid < 128) REDUCE(128) __syncthreads();
#endif
#if (THREADS_PER_BLOCK >= 128)
  if (tid < 64) REDUCE(64) __syncthreads();
#endif
  if (tid < 32) {
    if (THREADS_PER_BLOCK >= 64) REDUCE(32)
    if (THREADS_PER_BLOCK >= 32) REDUCE(16)
    if (THREADS_PER_BLOCK >= 16) REDUCE(8)
    if (THREADS_PER_BLOCK >= 8) REDUCE(4)
    if (THREADS_PER_BLOCK >= 4) REDUCE(2)
    if (THREADS_PER_BLOCK >= 2) REDUCE(1)
  }
  if (tid == 0) { index[blockIdx.x] = idxs[0]; value[blockIdx.x] = vals[0]; }
}
'''

LOAD = 'data[tid + %(x)d] = (i + %(x)d < n) ? signal[i + %(x)d] : 0;\n'

ACC = ('acc += data[tid + %(x)d + FILTER_X * %(y)d] * '
       'FILTER[%(x)d + FILTER_X * %(y)d];\n')

class Codebook(codebook.Codebook):
    def __init__(self, num_filters, filter_shape):
        if isinstance(filter_shape, int):
            filter_shape = (filter_shape, )
        if isinstance(filter_shape, list) or callable(filter_shape, 'next'):
            filter_shape = tuple(filter_shape)
        while len(filter_shape) < 3:
            filter_shape += (1, )

        try:
            x, y, z = filter_shape
        except:
            raise IndexError('CUDA pursuit not available for > 3 dimensions')

        kwargs = tuple(dict(x=i, y=j, z=k)
                       for i in range(x)
                       for j in range(y)
                       for k in range(z))

        load = ''.join(LOAD % kw for kw in kwargs)
        add = ''.join(ACC % kw for kw in kwargs)
        self._module = pycuda.compiler.SourceModule(SOURCE % (
            THREADS_PER_BLOCK, 'double', 'long', x, y, z, load, add))

        self._correlate = self._module.get_function('correlate')
        self._correlate.prepare('PPiii', (THREADS_PER_BLOCK, 1, 1))

        self._argmax = self._module.get_function('argmax')
        self._argmax.prepare('PPPi', (THREADS_PER_BLOCK, 1, 1))

        self._filter = self._module.get_global('FILTER')[0]

        super(Codebook, self).__init__(num_filters, filter_shape)

    def encode(self, signal, min_coeff=0., max_num_coeffs=-1, noise=0.):
        while len(signal.shape) < 3:
            signal.shape += (1, )

        assert len(signal.shape) == len(shape), \
            'signal %r, filter %r: ndim mismatch' % (signal.shape, shape)

        signal = pycuda.gpuarray.to_gpu(signal)

        grid = (int(numpy.ceil(float(signal.size) / THREADS_PER_BLOCK)), 1)

        indices = pycuda.gpuarray.zeros((1, ), int)
        coeffs = pycuda.gpuarray.zeros((1, ), float)
        scores = pycuda.gpuarray.zeros(
            (len(self.filters), ) +
            tuple(numpy.array(signal.shape) - numpy.array(shape) + 1),
            float)

        for i, f in enumerate(self.filters):
            cuda.memcpy_htod(self._filter, f)
            self._correlate(grid, scores[i:i+1], signal, *signal.shape)

        amplitude = self._abssum(signal)
        while max_num_coeffs != 0:
            max_num_coeffs -= 1

            self._argmax((1, 1), indices, coeffs, scores, scores.size)
            index = indices.get()[0]
            coeff = coeffs.get()[0]

            if coeff < min_coeff:
                logging.debug('halting: coefficient %d is %.2f < %.2f',
                              -max_num_coeffs, coeff, min_coeff)
                break

            whence = numpy.unravel_index(index, scores.shape)
            index, offset = whence[0], whence[1:]

            region = tuple(slice(o, o+shape[i]) for i, o in enumerate(offset))

            a = amplitude - self._abssum(signal[region])
            signal[region] -= coeff * self.filters[index]
            a += self._abssum(signal[region])

            if a > 1.01 * amplitude:
                logging.debug('halting: coefficient %.2f, filter %d, '
                              'offset %r yields amplitude %.2f > 1.01 * %.2f',
                              coeff, index, offset, a, amplitude)
                break
            amplitude = a

            for i, f in enumerate(self.filters):
                cuda.memcpy_htod(self._filter, f)
                self._correlate(grid, scores[i:i+1], signal, *signal.shape)

            yield index, coeff, offset

        else:
            logging.debug(
                'halting: final coefficient %d is %.2f', -max_num_coeffs, coeff)


class Trainer(codebook.Trainer):
    def _resize(self, *args, **kwargs):
        raise NotImplementedError

