# py-pursuit

This Python package contains some variants of the matching pursuit sparse coding
algorithm. Matching pursuit uses a set of "basis functions" or "codebook
filters" to greedily encode a raw signal in terms of a weighted sum of filters.
Using gradient ascent on the likelihood of an observed dataset, we can also
infer a likely set of filters from an unlabeled dataset of signals.

## Installing

Just use the setup.py script :

    python setup.py install

Or use pip and virtualenv for even more installation goodness :

    pip install lmj.pursuit

After installation, you can use the package by importing lmj.pursuit.

## Testing

The source distribution includes two primary tests: sound and image. The sound
test uses an experimental implementation that runs on a [CUDA][]-enabled
graphics device to encode sound waveforms. The image test encodes image pixels.

[CUDA]: http://www.nvidia.com/object/cuda_home_new.html

### Sound (CUDA)

The sound test runs matching pursuit on your 64-bit graphics card. To get
started, install py-cuda :

    pip install PyCUDA

Currently, this test requires that your graphics device support 64-bit floating
point values. If your graphics device is limited to 32-bit floats, you can add
`bit_depth=32` to the CudaPursuit constructor in the test.

The pursuit algorithm is trained using a sound waveform and reports the error
after encoding and decoding a test sound -- smaller numbers are better. Run this
test with :

    python test/sound.py

The general takeaway is that signal reproduction tends to improve with more
training (successive numbers within a group), with more codebook filters (first
column), and by using the multiple-frame (temporal) encoder instead of the
single-frame (standard) encoder. Interestingly, the standard encoder tends to do
worse with larger filters (second column), while the temporal encoder tends to
do worse with smaller filters.

If you have matplotlib installed, you can also save plots of the codebook
vectors during training by setting `GRAPHS = '/tmp/pursuit'` (or some other
directory name) in `test/sound.py`. Graphing doubles the test runtime, but
produces some pretty training artifacts.

### Image

You'll need to install [glumpy][] to run this test :

    pip install glumpy

The image test simply requires some image data to run :

    python test/images.py /path/to/my/image.jpg /path/to/another/image.png

You'll see a window appear on your desktop ; this window is divided into four
quadrants. At the upper-left is an image to be encoded. On the upper-right is
the reconstructed image. In the lower-left are the codebook filters being used
to perform the encoding. In the lower-right are the "feature maps" that show
where each codebook filter has been used to reconstruct the source image.

[glumpy]: http://code.google.com/p/glumpy/

## License

(The MIT License)

Copyright (c) 2010 Leif Johnson <leif@leifjohnson.net>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the 'Software'), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
