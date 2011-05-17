#include <Python.h>
#include <numpy/arrayobject.h>

static char err[200];

/* void correlate1d(1D-array signal, 1D-array filter, 1D-array *result)
 *
 * Compute the correlation of a 1D signal of shape (s, ) with a 1D filter of
 * shape (f, ). Stores results in the given 1D array of length s - f + 1.
 *
 * About 50x faster than scipy.signal.correlate(signal, filter, 'valid').
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

    if (result->nd > 1) {
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
    const int flens = filter->strides[0] / sizeof(double);
    const int flen = filter->dimensions[0];

    double *s = (double *) signal->data;
    double *f = (double *) filter->data;
    double *r = (double *) result->data;
    int i, j;
    double acc;
    for (i = 0; i < result->dimensions[0]; ++i) {
        acc = 0.0;
        for (j = 0; j < flen; ++j)
            acc += f[j * flens] * s[(i + j) * slens];
        r[i] = acc;
    }

    Py_RETURN_NONE;
}

/* void correlate2d(2D-array signal, 2D-array filter, 1D-array *result)
 *
 * Compute the correlation of a 2D signal of shape (s, k) with a 2D filter of
 * shape (f, k), where the second dimension in the signal equals the second
 * dimension in the filter. Stores results in the given 1D array of length
 * s - f + 1.
 *
 * About 50x faster than scipy.signal.correlate(signal, filter, 'valid').
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

    if (result->nd > 1) {
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
    const int flen = filter->dimensions[0];
    const int fwid = filter->dimensions[1];

    double *s = (double *) signal->data;
    double *f = (double *) filter->data;
    double *r = (double *) result->data;
    int i, j, k;
    double acc;
    for (i = 0; i < result->dimensions[0]; ++i) {
        acc = 0.0;
        for (j = 0; j < flen; ++j)
            for (k = 0; k < fwid; ++k)
                acc += f[j * flens + k * fwids] * s[(i + j) * slens + k * swids];
        r[i] = acc;
    }

    Py_RETURN_NONE;
}

static PyMethodDef PursuitMethods[] = {
    {"correlate1d", correlate1d, METH_VARARGS,
     "Correlate a 1D signal of shape (s, ) with a 1D filter of shape (f, )."},
    {"correlate2d", correlate2d, METH_VARARGS,
     "Correlate a 2D signal of shape (s, k) with a 2D filter of shape (f, k)."},
    {NULL, NULL, 0, NULL} /* sentinel */
};

PyMODINIT_FUNC
init_pursuit(void) {
    import_array();
    (void) Py_InitModule("_pursuit", PursuitMethods);
}
