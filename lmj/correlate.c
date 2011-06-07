/*
 * Copyright (c) 2011 Leif Johnson <leif@leifjohnson.net>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Implementations of double-precision signal correlation functions.
 *
 * These functions do a pointwise correlation between some signal s and a filter
 * f, storing the result in r. The correlation functions are specialized for a
 * small number of common cases, namely 1d and 2d arrays consisting of
 * double-precision values. The function signature is always
 *
 *   correlateX(s, f, r)
 *
 * The functions here are intended to be a replacement for equivalent calls to
 * scipy.signal.correlate(signal, filter, 'valid'). They provide a speedup of
 * approximately 10x over the scipy implementation.
 */

#include <Python.h>
#include <numpy/arrayobject.h>

static char err[200];

/* void correlate1d(1D-array signal, 1D-array filter, 1D-array *result)
 *
 * Compute the correlation of a 1D signal of shape (s, ) with a 1D filter of
 * shape (f, ). Stores results in the given 1D array of length s - f + 1.
 */
static PyObject *correlate1d(PyObject *self, PyObject *args) {
    PyArrayObject *signal, *filter, *result;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PyArray_Type, &signal,
                          &PyArray_Type, &filter,
                          &PyArray_Type, &result)) {
        PyErr_SetString(PyExc_ValueError, "correlate requires three ndarray arguments");
        return NULL;
    }
    if (signal->nd != 1) {
        PyErr_SetString(PyExc_ValueError, "signal must be 1D");
        return NULL;
    }
    if (filter->nd != 1) {
        PyErr_SetString(PyExc_ValueError, "filter must be 1D");
        return NULL;
    }
    if (result->nd != 1) {
        PyErr_SetString(PyExc_ValueError, "result must be 1D");
        return NULL;
    }
    if (PyArray_DIM(result, 0) != PyArray_DIM(signal, 0) - PyArray_DIM(filter, 0) + 1) {
        sprintf(err, "%d: result length must be %d - %d + 1",
                (int) PyArray_DIM(result, 0),
                (int) PyArray_DIM(signal, 0),
                (int) PyArray_DIM(filter, 0));
        PyErr_SetString(PyExc_ValueError, err);
        return NULL;
    }

    const int slens = signal->strides[0] / sizeof(double);
    const int flens = filter->strides[0] / sizeof(double);
    const int flend = filter->dimensions[0];
    const int rlend = result->dimensions[0];

    double *s = (double *) signal->data;
    double *f = (double *) filter->data;
    double *r = (double *) result->data;
    int i, a;
    double acc;
    for (i = 0; i < rlend; ++i) {
        acc = 0.0;
        for (a = 0; a < flend; ++a)
            acc += f[a * flens] * s[(i + a) * slens];
        r[i] = acc;
    }

    Py_RETURN_NONE;
}

/* void correlate1d_from_2d(2D-array signal, 2D-array filter, 1D-array *result)
 *
 * Compute the correlation of a 2D signal of shape (s, k) with a 2D filter of
 * shape (f, k), where the second dimension in the signal equals the second
 * dimension in the filter. Stores results in the given 1D array of length
 * s - f + 1.
 */
static PyObject *correlate1d_from_2d(PyObject *self, PyObject *args) {
    PyArrayObject *signal, *filter, *result;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PyArray_Type, &signal,
                          &PyArray_Type, &filter,
                          &PyArray_Type, &result)) {
        PyErr_SetString(PyExc_ValueError, "correlate requires three ndarray arguments");
        return NULL;
    }
    if (signal->nd != 2) {
        PyErr_SetString(PyExc_ValueError, "signal must be 2D");
        return NULL;
    }
    if (filter->nd != 2) {
        PyErr_SetString(PyExc_ValueError, "filter must be 2D");
        return NULL;
    }
    if (result->nd != 1) {
        PyErr_SetString(PyExc_ValueError, "result must be 1D");
        return NULL;
    }
    if (PyArray_DIM(result, 0) != PyArray_DIM(signal, 0) - PyArray_DIM(filter, 0) + 1) {
        sprintf(err, "%d: result length must be %d - %d + 1",
                (int) PyArray_DIM(result, 0),
                (int) PyArray_DIM(signal, 0),
                (int) PyArray_DIM(filter, 0));
        PyErr_SetString(PyExc_ValueError, err);
        return NULL;
    }
    if (PyArray_DIM(signal, 1) != PyArray_DIM(filter, 1)) {
        sprintf(err, "signal channels (%d) does not match filter channels (%d)",
                (int) PyArray_DIM(signal, 1),
                (int) PyArray_DIM(filter, 1));
        PyErr_SetString(PyExc_ValueError, err);
        return NULL;
    }

    const int slens = signal->strides[0] / sizeof(double);
    const int swids = signal->strides[1] / sizeof(double);

    const int flens = filter->strides[0] / sizeof(double);
    const int fwids = filter->strides[1] / sizeof(double);

    const int flend = filter->dimensions[0];
    const int fwidd = filter->dimensions[1];

    const int rlend = result->dimensions[0];

    double *s = (double *) signal->data;
    double *f = (double *) filter->data;
    double *r = (double *) result->data;
    int i, a, b;
    double acc;
    for (i = 0; i < rlend; ++i) {
        acc = 0.0;
        for (a = 0; a < flend; ++a)
            for (b = 0; b < fwidd; ++b)
                acc += f[a * flens + b * fwids] * s[(i + a) * slens + b * swids];
        r[i] = acc;
    }

    Py_RETURN_NONE;
}

/* void correlate2d(2D-array signal, 2D-array filter, 2D-array *result)
 *
 * Compute the correlation of a 2D signal of shape (s, t) with a 2D filter of
 * shape (f, g). Stores results in the given 2D array of shape
 * (s - f + 1, t - g + 1).
 */
static PyObject *correlate2d(PyObject *self, PyObject *args) {
    PyArrayObject *signal, *filter, *result;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PyArray_Type, &signal,
                          &PyArray_Type, &filter,
                          &PyArray_Type, &result)) {
        PyErr_SetString(PyExc_ValueError, "correlate requires three ndarray arguments");
        return NULL;
    }
    if (signal->nd != 2) {
        PyErr_SetString(PyExc_ValueError, "signal must be 2D");
        return NULL;
    }
    if (filter->nd != 2) {
        PyErr_SetString(PyExc_ValueError, "filter must be 2D");
        return NULL;
    }
    if (result->nd != 2) {
        PyErr_SetString(PyExc_ValueError, "result must be 2D");
        return NULL;
    }
    if ((PyArray_DIM(result, 0) != PyArray_DIM(signal, 0) - PyArray_DIM(filter, 0) + 1) ||
        (PyArray_DIM(result, 1) != PyArray_DIM(signal, 1) - PyArray_DIM(filter, 1) + 1)) {
        sprintf(err, "(%d, %d): result shape must be (%d - %d + 1, %d - %d + 1)",
                (int) PyArray_DIM(result, 0),
                (int) PyArray_DIM(result, 1),
                (int) PyArray_DIM(signal, 0),
                (int) PyArray_DIM(filter, 0),
                (int) PyArray_DIM(signal, 1),
                (int) PyArray_DIM(filter, 1));
        PyErr_SetString(PyExc_ValueError, err);
        return NULL;
    }

    const int slens = signal->strides[0] / sizeof(double);
    const int swids = signal->strides[1] / sizeof(double);

    const int flens = filter->strides[0] / sizeof(double);
    const int fwids = filter->strides[1] / sizeof(double);

    const int rlens = result->strides[0] / sizeof(double);
    const int rwids = result->strides[1] / sizeof(double);

    const int flend = filter->dimensions[0];
    const int fwidd = filter->dimensions[1];

    const int rlend = result->dimensions[0];

    double *s = (double *) signal->data;
    double *f = (double *) filter->data;
    double *r = (double *) result->data;
    int i, j, a, b;
    double acc;
    for (i = 0; i < rlend; ++i) {
        for (j = 0; j < result->dimensions[1]; ++j) {
            acc = 0.0;
            for (a = 0; a < flend; ++a)
                for (b = 0; b < fwidd; ++b)
                    acc += f[a * flens + b * fwids] *
                        s[(i + a) * slens + (j + b) * swids];
            r[i * rlens + j * rwids] = acc;
        }
    }

    Py_RETURN_NONE;
}

/* void correlate2d_from_rgb(3D-array signal, 3D-array filter, 2D-array *result)
 *
 * Compute the correlation of a 3D signal of shape (s, t, 3) with a 3D filter of
 * shape (f, g, 3). Stores results in the given 2D array of shape
 * (s - f + 1, t - g + 1).
 */
static PyObject *correlate2d_from_rgb(PyObject *self, PyObject *args) {
    PyArrayObject *signal, *filter, *result;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PyArray_Type, &signal,
                          &PyArray_Type, &filter,
                          &PyArray_Type, &result)) {
        PyErr_SetString(PyExc_ValueError, "correlate requires three ndarray arguments");
        return NULL;
    }
    if (signal->nd != 3) {
        PyErr_SetString(PyExc_ValueError, "signal must be 3D");
        return NULL;
    }
    if (filter->nd != 3) {
        PyErr_SetString(PyExc_ValueError, "filter must be 3D");
        return NULL;
    }
    if (result->nd != 2) {
        PyErr_SetString(PyExc_ValueError, "result must be 2D");
        return NULL;
    }
    if ((PyArray_DIM(result, 0) != PyArray_DIM(signal, 0) - PyArray_DIM(filter, 0) + 1) ||
        (PyArray_DIM(result, 1) != PyArray_DIM(signal, 1) - PyArray_DIM(filter, 1) + 1)) {
        sprintf(err, "(%d, %d): result shape must be (%d - %d + 1, %d - %d + 1)",
                (int) PyArray_DIM(result, 0),
                (int) PyArray_DIM(result, 1),
                (int) PyArray_DIM(signal, 0),
                (int) PyArray_DIM(filter, 0),
                (int) PyArray_DIM(signal, 1),
                (int) PyArray_DIM(filter, 1));
        PyErr_SetString(PyExc_ValueError, err);
        return NULL;
    }
    if (PyArray_DIM(signal, 2) != 3) {
        sprintf(err, "%d: signal channels must be 3 for RGB correlation",
                (int) PyArray_DIM(signal, 2));
        PyErr_SetString(PyExc_ValueError, err);
        return NULL;
    }
    if (PyArray_DIM(filter, 2) != 3) {
        sprintf(err, "%d: filter channels must be 3 for RGB correlation",
                (int) PyArray_DIM(filter, 2));
        PyErr_SetString(PyExc_ValueError, err);
        return NULL;
    }

    const int slens = signal->strides[0] / sizeof(double);
    const int swids = signal->strides[1] / sizeof(double);
    const int sdeps = signal->strides[2] / sizeof(double);

    const int flens = filter->strides[0] / sizeof(double);
    const int fwids = filter->strides[1] / sizeof(double);
    const int fdeps = filter->strides[2] / sizeof(double);

    const int rlens = result->strides[0] / sizeof(double);
    const int rwids = result->strides[1] / sizeof(double);

    const int flend = filter->dimensions[0];
    const int fwidd = filter->dimensions[1];

    const int rlend = result->dimensions[0];

    /* TODO: see if we can incorporate a perceptually motivated "distance" for
     * RGB like this one from http://www.compuphase.com/cmetric.htm :
     *
     *   r = (c1[r] + c2[r]) / 2
     *   d = c1 - c2
     *   dist = sqrt((2 + r / 256) * d[r] * d[r] +
     *               4 * d[g] * d[g] +
     *               (2 + (255 - r) / 256) * d[b] * d[b])
     */
    double *s = (double *) signal->data;
    double *f = (double *) filter->data;
    double *r = (double *) result->data;
    int i, j, a, b;
    double acc;
    for (i = 0; i < rlend; ++i) {
        for (j = 0; j < result->dimensions[1]; ++j) {
            acc = 0.0;
            for (a = 0; a < flend; ++a)
                for (b = 0; b < fwidd; ++b)
                    acc +=
                        f[a * flens + b * fwids] *
                          s[(i + a) * slens + (j + b) * swids] +
                        f[a * flens + b * fwids + fdeps] *
                          s[(i + a) * slens + (j + b) * swids + sdeps] +
                        f[a * flens + b * fwids + 2 * fdeps] *
                          s[(i + a) * slens + (j + b) * swids + 2 * sdeps];
            r[i * rlens + j * rwids] = acc;
        }
    }

    Py_RETURN_NONE;
}

static PyMethodDef CorrelateMethods[] = {
    {"correlate1d", correlate1d, METH_VARARGS,
     "Correlate a 1D signal of length s with a 1D filter of length f."},
    {"correlate1d_from_2d", correlate1d_from_2d, METH_VARARGS,
     "Correlate a 2D signal of shape (s, k) with a 2D filter of shape (f, k)."},
    {"correlate2d", correlate2d, METH_VARARGS,
     "Correlate a 2D signal of shape (s, t) with a 2D filter of shape (f, g)."},
    {"correlate2d_from_rgb", correlate2d_from_rgb, METH_VARARGS,
     "Correlate a 3D signal of shape (s, t, 3) with a 3D filter of shape (f, g, 3)."},
    {NULL, NULL, 0, NULL} /* sentinel */
};

PyMODINIT_FUNC
init_correlate(void) {
    import_array();
    (void) Py_InitModule("_correlate", CorrelateMethods);
}
