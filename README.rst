Matching Pursuit
================

This Python package contains two variants of the matching pursuit sparse coding
algorithm. Matching pursuit uses an overcomplete set of "basis functions" or
"codebook filters" to greedily encode a raw signal in terms of a weighted sum of
filters. Using gradient ascent we can also infer a likely set of filters from an
unlabeled dataset of signals.

Building
--------

You'll need to have the numpy headers installed to build the C module (which
speeds up the algorithm by about a factor of 2). After you've installed those,
just do the usual setup.py dance ::

  python setup.py build
  ln -s ../build/lib.*/lmj/_pursuit.so lmj

The second step is needed if you want to use the package from the current
working directory (e.g., if you want to run the test -- see below). If you're
just going to install the package (see below), ignore the symlink.

Testing
-------

The source distribution includes a basic test module that runs matching pursuit
on a training sound and reports the error after encoding and decoding a test
sound -- smaller numbers are better. Run this test with ::

  python pursuit_test.py

from the base directory in the source tree. On my machine, the test takes
between 1 and 5 minutes to run to completion, but you can stop it at any time.

The general takeaway is that signal reproduction tends to improve with more
training (rows of numbers), with more codebook filters (first column), and by
using the multiple-frame encoder instead of the single-frame encoder.
Interestingly, the single-frame encoder does worse with larger filters (second
column), while the multiple-frame encoder tends to do worse with smaller
filters.

If you have matplotlib installed, you can also generate plots of the codebook
vectors during training (stored in /tmp/pursuit) by changing GRAPHS = True in
pursuit_test.py. This doubles the test runtime, but produces some cool graphs.

Installing
----------

Again, use the setup.py script ::

  python setup.py install

Use pip and virtualenv for even more installation goodness !

After installation, you can use the package by importing lmj.pursuit.

Also, have fun !
