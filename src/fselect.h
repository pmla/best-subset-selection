#ifndef FSELECT_H
#define FSELECT_H

#ifdef __cplusplus
extern "C" {
#endif

void fselect(int m, int n, double* A, double* b, int* features, double* weights);

#ifdef __cplusplus
}
#endif

#endif
