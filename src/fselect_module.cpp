#include <Python.h>
#include <ndarraytypes.h>
#include <arrayobject.h>
#include "fselect.h"
#include <cstring>
#include <cassert>
#include <vector>


#ifdef __cplusplus
extern "C" {
#endif

static PyObject* error(PyObject* type, const char* msg)
{
	PyErr_SetString(type, msg);
	return NULL;
}

static PyObject* solve_greedy(PyObject* self, PyObject* args, PyObject* kwargs)
{
	(void)self;
	PyObject* obj_A = NULL;
	PyObject* obj_b = NULL;
	PyObject* obj_Acont = NULL;
	PyObject* obj_bcont = NULL;

	if (!PyArg_ParseTuple(args, "OO", &obj_A, &obj_b))
		return NULL;

	obj_Acont = PyArray_ContiguousFromAny(obj_A, NPY_DOUBLE, 1, 2);
	if (obj_Acont == NULL)
		return error(PyExc_TypeError, "Invalid input data: A");

	obj_bcont = PyArray_ContiguousFromAny(obj_b, NPY_DOUBLE, 1, 1);
	if (obj_bcont == NULL)
		return error(PyExc_TypeError, "Invalid input data: c");

	if (PyArray_NDIM(obj_Acont) != 2)
		return error(PyExc_TypeError, "A must be two-dimensional");

	if (PyArray_NDIM(obj_bcont) != 1)
		return error(PyExc_TypeError, "c must be one-dimensional");

	if (PyArray_DIM(obj_Acont, 0) != PyArray_DIM(obj_bcont, 0))
		return error(PyExc_TypeError, "shape mismatch between A and b");

	int m = PyArray_DIM(obj_Acont, 0);
	int n = PyArray_DIM(obj_Acont, 1);
	double* A = (double*)PyArray_DATA((PyArrayObject*)obj_A);
	double* b = (double*)PyArray_DATA((PyArrayObject*)obj_b);

	npy_intp dim[2] = {n, n};
	PyObject* obj_features = PyArray_SimpleNew(1, dim, NPY_INT);
	int* features = (int*)PyArray_DATA((PyArrayObject*)obj_features);
	memset(features, 0, n * sizeof(int));

	PyObject* obj_weights = PyArray_SimpleNew(2, dim, NPY_DOUBLE);
	double* weights = (double*)PyArray_DATA((PyArrayObject*)obj_weights);
	memset(weights, 0, n * n * sizeof(double));

	fselect(m, n, A, b, features, weights);

	PyObject* result = Py_BuildValue("OO", obj_features, obj_weights);
	Py_DECREF(obj_Acont);
	Py_DECREF(obj_bcont);
	Py_DECREF(obj_features);
	Py_DECREF(obj_weights);
	return result;
}

static PyMethodDef _fselect_methods[] = {
	{
		"solve_greedy",
		(PyCFunction)solve_greedy,
		METH_VARARGS,
		"Description."
	},
	{NULL}
};

static struct PyModuleDef _fselect_definition = {
	PyModuleDef_HEAD_INIT,
	"_fselect",
	"Greedy algorithm for best subset selection.",
	-1,
	_fselect_methods,
	NULL,
	NULL,
	NULL,
	NULL,
};

PyMODINIT_FUNC PyInit__fselect(void)
{
	Py_Initialize();
	import_array();
	return PyModule_Create(&_fselect_definition);
}

#ifdef __cplusplus
}
#endif

