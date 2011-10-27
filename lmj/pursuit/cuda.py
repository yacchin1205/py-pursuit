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

import sys
import time
import numpy
import logging
import pycuda.autoinit
import pycuda.driver
import pycuda.compiler
import pycuda.gpuarray
import pycuda.reduction

import correlation as codebook

SOURCE = '''
#define THREADS_PER_BLOCK %d

typedef %s float_;

#define FILTER_1 %d
#define FILTER_2 %d
#define FILTER_3 %d

__device__ __constant__ float_ FILTER[FILTER_1 * FILTER_2 * FILTER_3];

__global__ void correlate(
    float_ *result,
    const float_ *signal,
    const int signal_1,
    const int signal_2,
    const int signal_3)
{
  const int result_1 = signal_1 - FILTER_1 + 1;
  const int result_2 = signal_2 - FILTER_2 + 1;
  const int result_3 = signal_3 - FILTER_3 + 1;

  // calculate the indices along all axes of the output (target) memory.
  const int target = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  const int target_1 = ((target / result_3) / result_2) %% result_1;
  const int target_2 = (target / result_3) %% result_2;
  const int target_3 = target %% result_3;

  // use those indices to find the source memory for the correlation.
  const int source = target_3 + signal_3 * (target_2 + signal_2 * target_1);

  if (target < result_1 * result_2 * result_3) {
    float_ acc = 0;
    for (int i_1 = 0; i_1 < FILTER_1; ++i_1)
      for (int i_2 = 0; i_2 < FILTER_2; ++i_2)
        for (int i_3 = 0; i_3 < FILTER_3; ++i_3)
          acc += signal[source + i_3 + signal_3 * (i_2 + signal_2 * i_1)] *
              FILTER[i_3 + FILTER_3 * (i_2 + FILTER_2 * i_1)];
    result[target] = acc;
  }
}

#define REDUCE(i) { if (vals[tid] <= vals[tid + i]) { vals[tid] = vals[tid + i]; idxs[tid] = idxs[tid + i]; } }

__global__ void argmax(
    float_ *result,
    const float_ *data,
    const int n)
{
  const int tid = threadIdx.x;
  __shared__ float_ vals[THREADS_PER_BLOCK];
  __shared__ int idxs[THREADS_PER_BLOCK];
  float_ val = (tid < n) ? data[tid] : data[0] + 1;
  int idx = (tid < n) ? tid : -1;
  for (int i = tid; i < n; i += THREADS_PER_BLOCK)
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
  if (tid == 0) { result[0] = (float_) idxs[0]; result[1] = vals[0]; }
}

__global__ void subtract(
    float_ *signal,
    float_ *indices,
    float_ *filters,
    const int signal_1,
    const int signal_2,
    const int signal_3,
    const int result_1,
    const int result_2,
    const int result_3)
{
  const float_ coeff = indices[1];
  const int index = (int) indices[0];
  const int filter = index / result_3 / result_2 / result_1;
  const int target_1 = ((index / result_3) / result_2) %% result_1;
  const int target_2 = (index / result_3) %% result_2;
  const int target_3 = index %% result_3;
  const int source = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (source < FILTER_1 * FILTER_2 * FILTER_3)
    signal[source + target_3 + signal_3 * (target_2 + signal_2 * target_1)] -=
        coeff * filters[filter * FILTER_1 * FILTER_2 * FILTER_3 + source];
}

'''


class Codebook(codebook.Codebook):
    def __init__(self, num_filters, filter_shape, bit_depth=64, threads_per_block=512):
        if isinstance(filter_shape, int):
            filter_shape = (filter_shape, )

        super(Codebook, self).__init__(num_filters, filter_shape)

        i = j = k = 1
        if len(filter_shape) > 0:
            i = filter_shape[0]
        if len(filter_shape) > 1:
            j = filter_shape[1]
        if len(filter_shape) > 2:
            k = filter_shape[2]
        if len(filter_shape) > 3:
            raise IndexError('CUDA pursuit not available for > 3 dimensions')

        self._block = (threads_per_block, 1, 1)
        self._float_type = {64: numpy.float64, 32: numpy.float32}[bit_depth]
        float_name = {64: 'double', 32: 'float'}[bit_depth]

        source = SOURCE % (threads_per_block, float_name, i, j, k)
        self._module = pycuda.compiler.SourceModule(source)
        self._filter = self._module.get_global('FILTER')[0]
        self._correlate = self._module.get_function('correlate')
        self._argmax = self._module.get_function('argmax')
        self._subtract = self._module.get_function('subtract')

    def encode(self, signal, min_coeff=0., max_num_coeffs=-1, minibatch=1000):
        '''Generate a set of codebook coefficients for encoding a signal.

        signal: A signal to encode.
        min_coeff: Stop encoding when the magnitude of coefficients falls below
          this threshold. Use 0 to encode until max_num_coeffs is reached.
        max_num_coeffs: Stop encoding when we have generated this many
          coefficients. Use a negative value to encode until min_coeff is
          reached.
        minibatch: Extract this many coefficients at once on the GPU.

        This method generates a sequence of tuples of the form (index,
        coefficient, offset), where index refers to a codebook filter and
        coefficient is the scalar multiple of the filter that is present in the
        input signal starting at the given offsets.

        See the Trainer class for an example of how to use these results to
        update the codebook filters.
        '''
        ndims = len(signal.shape)

        # copy the filters and signal to the gpu.
        gpu_signal = pycuda.gpuarray.to_gpu(signal.astype(self._float_type))
        gpu_filters = pycuda.gpuarray.to_gpu(
            numpy.array(self.filters).astype(self._float_type))

        # reserve some memory on the gpu for holding intermediate results.
        shape = tuple(ss - fs + 1 for ss, fs in zip(signal.shape, self.filters[0].shape))
        while len(shape) < 3:
            shape += (1, )
        indices = pycuda.gpuarray.zeros((2 * minibatch, ), self._float_type)
        scores = pycuda.gpuarray.zeros(
            (len(self.filters), ) + tuple(shape), self._float_type)

        # calculate encoding coefficients in a large chunk on the gpu, and then
        # use those coefficients to simulate the encoding on the cpu (by
        # progressively subtracting codebook filters from the signal in main
        # memory).
        processing = True
        while processing:
            self._encode(gpu_signal, gpu_filters, indices, scores, minibatch)
            idx = indices.get()
            for b in range(minibatch):
                max_num_coeffs -= 1
                if max_num_coeffs == 0:
                    processing = False
                    break

                index, coeff = idx[2 * b:2 * (b + 1)]
                if coeff < min_coeff:
                    logging.debug('halting: coefficient %d is %.2f < %.2f',
                                  -max_num_coeffs, coeff, min_coeff)
                    processing = False
                    break

                # subtract the filter from the signal in main memory.
                whence = numpy.unravel_index(int(index), scores.shape)
                index, offset = whence[0], whence[1:]
                f = self.filters[index]
                region = tuple(slice(o, o + fs) for o, fs in zip(offset, f.shape))
                signal[region] -= coeff * f

                yield index, coeff, offset[:ndims]

        logging.debug(
            'halting: final coefficient %d is %.2f', -max_num_coeffs, coeff)

    def _encode(self, signal, filters, indices, scores, minibatch):
        tpb = self._block[0]
        correlate_grid = int(numpy.ceil(float(scores.size) / tpb))
        argmax_grid = 1
        subtract_grid = int(numpy.ceil(float(filters[0].size) / tpb))

        signal_shape = tuple(numpy.int32(s) for s in signal.shape)
        while len(signal_shape) < 3:
            signal_shape += (numpy.int32(1), )

        score_shape = tuple(numpy.int32(s) for s in scores.shape)

        for b in range(minibatch):
            for i in range(filters.shape[0]):
                f = filters[i]
                pycuda.driver.memcpy_dtod(self._filter, f.gpudata, f.nbytes)
                self._correlate(scores[i], signal, *signal_shape,
                                block=self._block, grid=(correlate_grid, 1))
            self._argmax(indices[2 * b:], scores, numpy.int32(scores.size),
                         block=self._block, grid=(argmax_grid, 1))
            self._subtract(signal, indices[2 * b:], filters,
                           *(signal_shape + score_shape[1:]),
                           block=self._block, grid=(subtract_grid, 1))


class Trainer(codebook.Trainer):
    def _resize(self, *args, **kwargs):
        raise NotImplementedError



if __name__ == '__main__':
    import time
    import scipy.signal

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    s = numpy.random.randn(301, 307, 1)
    f = numpy.random.randn(16, 17, 1)

    c = Codebook(10, f.shape)

    scores = pycuda.gpuarray.zeros(
        tuple(a - b + 1 for a, b in zip(s.shape, f.shape)), c._float_type)
    indices = pycuda.gpuarray.zeros((2, ), c._float_type)

    grid = (int(numpy.ceil(float(scores.size) / 512)), 1)
    block = (512, 1, 1)
    logging.info('correlating with %r blocks of %r threads', grid, block)

    start = time.time()
    pycuda.driver.memcpy_htod(c._filter, f.astype(c._float_type))
    c._correlate(scores, pycuda.gpuarray.to_gpu(s.astype(c._float_type)),
                 *tuple(numpy.int32(x) for x in s.shape),
                 grid=grid, block=block)
    compute = time.time()
    c._argmax(indices, scores, numpy.int32(scores.size), grid=(1, 1), block=block)
    argmax = time.time()
    indices.get()
    logging.info('cuda takes %dms for correlation, %dms for argmax, %dms total',
                 1000 * (compute - start),
                 1000 * (argmax - start),
                 1000 * (time.time() - start))

    start = time.time()
    sr = scipy.signal.correlate(s, f, 'valid')
    compute = time.time()
    id = sr.argmax()
    logging.info('scipy takes %dms for correlation, %dms total',
                 1000 * (compute - start),
                 1000 * (time.time() - start))

    logging.info('cuda argmax: %r, scipy argmax: %r', indices.get(), sr.argmax())
    logging.info('error: %r', numpy.linalg.norm(sr - scores.get()))
    logging.info(numpy.allclose(sr, scores.get()) and
                 'they match !' or 'fix some bugs.')
