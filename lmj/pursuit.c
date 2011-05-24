#include <Python.h>
#include <numpy/arrayobject.h>

static char err[200];

/* void correlate1d(1D-array signal, 1D-array filter, 1D-array *result)
 *
 * Compute the correlation of a 1D signal of shape (s, ) with a 1D filter of
 * shape (f, ). Stores results in the given 1D array of length s - f + 1.
 *
 * About Nx faster than scipy.signal.correlate(signal, filter, 'valid').
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

    double *s = (double *) signal->data;
    double *f = (double *) filter->data;
    double *r = (double *) result->data;
    int i, j;
    double acc;
    for (i = 0; i < result->dimensions[0]; ++i) {
        acc = 0.0;
        for (j = 0; j < flend; ++j)
            acc += f[j * flens] * s[(i + j) * slens];
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
 *
 * About 50x faster than scipy.signal.correlate(signal, filter, 'valid').
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

    if (signal->nd != 1) {
        PyErr_SetString(PyExc_ValueError, "signal must be 2D");
        return NULL;
    }
    if (filter->nd != 1) {
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
        sprintf(err, "signal width (%d) does not match filter width (%d)",
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

    double *s = (double *) signal->data;
    double *f = (double *) filter->data;
    double *r = (double *) result->data;
    int i, j, k;
    double acc;
    for (i = 0; i < result->dimensions[0]; ++i) {
        acc = 0.0;
        for (j = 0; j < flend; ++j)
            for (k = 0; k < fwidd; ++k)
                acc += f[j * flens + k * fwids] * s[(i + j) * slens + k * swids];
        r[i] = acc;
    }

    Py_RETURN_NONE;
}

/* void correlate2d(2D-array signal, 2D-array filter, 2D-array *result)
 *
 * Compute the correlation of a 2D signal of shape (s, t) with a 2D filter of
 * shape (f, g). Stores results in the given 2D array of shape
 * (s - f + 1, t - g + 1).
 *
 * About Nx faster than scipy.signal.correlate(signal, filter, 'valid').
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

    double *s = (double *) signal->data;
    double *f = (double *) filter->data;
    double *r = (double *) result->data;
    int i, j, a, b;
    double acc;
    for (i = 0; i < result->dimensions[0]; ++i) {
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
 *
 * About Nx faster than scipy.signal.correlate(signal, filter, 'valid').
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
    if (signal->dimensions[0] != 3) {
        PyErr_SetString(PyExc_ValueError, "signal must have 3 as the third dimension");
        return NULL;
    }
    if (filter->nd != 3) {
        PyErr_SetString(PyExc_ValueError, "filter must be 3D");
        return NULL;
    }
    if (filter->dimensions[0] != 3) {
        PyErr_SetString(PyExc_ValueError, "filter must have 3 as the third dimension");
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
    for (i = 0; i < result->dimensions[0]; ++i) {
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

static PyMethodDef PursuitMethods[] = {
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
init_pursuit(void) {
    import_array();
    (void) Py_InitModule("_pursuit", PursuitMethods);
}
