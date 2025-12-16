// PyCaller.h
#ifndef PYCALLER_H
#define PYCALLER_H
#include <Python.h>
PyObject *initialize_python();
void finalize_python();
void start_up_in_python(int init_flag, int epoch_start, double learning_rate);
void update_weights_in_python(double *point_input, double point_res, double dR_dE_numerical, int epoch, int save_flag);
void compute_E_from_python(double x1, double x2, double *E);
void change_learning_rate_in_python(double new_lr);


extern PyObject *pModule;  // Declaration of the global variable

#endif // PYCALLER_H
