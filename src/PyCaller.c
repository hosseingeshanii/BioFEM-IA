#include <Python.h>
#include  <petscvec.h>
#include  "variables.h"
#include "PyCaller.h"

extern PyObject *pModule;


PyObject *initialize_python() {
    // Initialize the Python interpreter
    Py_Initialize();

    // Load the Python module
    PyObject *pName = PyUnicode_DecodeFSDefault("ELASTICITY");  
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule == NULL) {
        PyErr_Print();
        Py_Finalize();
        return NULL;  // Return NULL if module import fails
    }
    PetscPrintf(PETSC_COMM_SELF, "Python module is initialized in initialize_python! pModule: %p\n", (void*)pModule);

    return pModule;  // Return the module object
}

void finalize_python() {
    if (pModule != NULL) {
        Py_DECREF(pModule);
    }
    
    Py_Finalize();
}

void compute_E_from_python(double x1, double x2, double *E) {
    //PetscPrintf(PETSC_COMM_SELF, "Check in compute_E_from_python!\n");

    // Check if pModule is initialized
    if (pModule == NULL) {
        PetscPrintf(PETSC_COMM_SELF, "Python module is not initialized!\n");
        return;
    }
    //PetscPrintf(PETSC_COMM_SELF, "Python module is initialized in compute_E_from_python! pModule: %p\n", (void*)pModule);

    PyObject *pFunc_compute_E = PyObject_GetAttrString(pModule, "compute_E");
    //PetscPrintf(PETSC_COMM_SELF, "Check after pFunc_compute_E!\n");
    if (pFunc_compute_E && PyCallable_Check(pFunc_compute_E)) {
        PyObject *pArgs = PyTuple_Pack(2, PyFloat_FromDouble(x1), PyFloat_FromDouble(x2));
        PyObject *pValue = PyObject_CallObject(pFunc_compute_E, pArgs);
        Py_DECREF(pArgs);

        if (pValue != NULL) {
            *E = PyFloat_AsDouble(pValue);
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
        Py_XDECREF(pFunc_compute_E);
    } else {
        if (PyErr_Occurred()) PyErr_Print();
        Py_XDECREF(pFunc_compute_E);
    }
}

void update_weights_in_python(double *point_input, double point_res, double dR_dE_numerical, int epoch, int save_flag) {
    PyObject *pFunc;
    PyObject *pArgs, *pValue;

    // Ensure the Python module is not NULL
    if (pModule == NULL) {
        PetscPrintf(PETSC_COMM_SELF, "Python module is NULL!\n");
        return;
    }
    
    // Get the Python function
    pFunc = PyObject_GetAttrString(pModule, "update_weights");
    if (pFunc && PyCallable_Check(pFunc)) {
        // Prepare the arguments

        PyObject *point_input_list = PyList_New(2);
        PyList_SetItem(point_input_list, 0, PyFloat_FromDouble(point_input[0]));
        PyList_SetItem(point_input_list, 1, PyFloat_FromDouble(point_input[1]));
        
        
        // Pack arguments        
        pArgs = PyTuple_Pack(5, 
                             point_input_list, 
                             PyFloat_FromDouble(point_res), 
                             PyFloat_FromDouble(dR_dE_numerical), 
                             PyLong_FromLong(epoch), 
                             PyLong_FromLong(save_flag));  // Pass save_flag as an integer (0 or 1)
        
        pValue = PyObject_CallObject(pFunc, pArgs);
        //PetscPrintf(PETSC_COMM_SELF, "CHECK after pValue \n");
        Py_DECREF(pArgs);

        if (pValue != NULL) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
        Py_XDECREF(pFunc);
    } else {
        if (PyErr_Occurred()) PyErr_Print();
        Py_XDECREF(pFunc);
    }
}

void start_up_in_python(int init_flag, int epoch_start, double learning_rate) {
    PyObject *pFunc;
    PyObject *pArgs, *pValue;

    // Ensure the Python module is not NULL
    if (pModule == NULL) {
        PetscPrintf(PETSC_COMM_SELF, "Python module is NULL!\n");
        return;
    }

    // Get the Python function
    pFunc = PyObject_GetAttrString(pModule, "start_up");
    if (pFunc && PyCallable_Check(pFunc)) {
        // Prepare the arguments
        PyObject *py_init_flag = PyLong_FromLong(init_flag);
        PyObject *py_epoch_start = PyLong_FromLong(epoch_start);
        PyObject *py_learning_rate = PyFloat_FromDouble(learning_rate);
        
        // Pack arguments
        pArgs = PyTuple_Pack(3, py_init_flag, py_epoch_start, py_learning_rate);

        // Call the Python function
        pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);

        if (pValue != NULL) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
        Py_XDECREF(pFunc);
    } else {
        if (PyErr_Occurred()) PyErr_Print();
        Py_XDECREF(pFunc);
    }
}


void change_learning_rate_in_python(double new_lr) {
    PyObject *pFunc;
    PyObject *pArgs, *pValue;

    // Ensure the Python module is not NULL
    if (pModule == NULL) {
        PetscPrintf(PETSC_COMM_SELF, "Python module is NULL!\n");
        return;
    }

    // Get the Python function
    pFunc = PyObject_GetAttrString(pModule, "update_learning_rate");
    if (pFunc && PyCallable_Check(pFunc)) {
        // Prepare the arguments
        pArgs = PyTuple_Pack(1, PyFloat_FromDouble(new_lr));

        // Call the Python function
        pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);

        if (pValue != NULL) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
        Py_XDECREF(pFunc);
    } else {
        if (PyErr_Occurred()) PyErr_Print();
        Py_XDECREF(pFunc);
    }
}
