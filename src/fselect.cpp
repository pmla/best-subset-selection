#include <cmath>
#include <cstring>
#include <vector>


static void cholesky_update(int n, double* L, double* A, int i) {

	for (int j=0;j<i;j++) {

		double p = 0;
		for (int k=0;k<j;k++) {
			p += L[i * n + k] * L[j * n + k];
		}

		L[i * n + j] = 1 / L[j * n + j] * (A[i * n + j] - p);
	}

	{
		double p = 0;
		for (int k=0;k<i;k++) {
			p += L[i * n + k] * L[i * n + k];
		}

		L[i * n + i] = sqrt(A[i * n + i] - p);
	}
}

static void forward_substitution(int n, int count, double* L, double* b, double* x) {

	for (int i=0;i<count;i++) {
		double p = 0;
		for (int j=0;j<i;j++)
			p += x[j] * L[i * n + j];

		x[i] = (b[i] - p) / L[i * n + i];
	}
}

static void backward_substitution(int n, int count, double* L, double* b, double* x) {

	for (int it0=0;it0<count;it0++) {
		int i = count - it0 - 1;

		double p = 0;
		for (int it1=0;it1<it0;it1++) {
			int j = count - it1 - 1;
			p += x[j] * L[j * n + i];
		}

		x[i] = (b[i] - p) / L[i * n + i];
	}
}

static void matrix_vector(int n, int count, double* A, double* b, double* x) {

	for (int i=0;i<count;i++) {

		double acc = 0;
		for (int j=0;j<count;j++) {
			acc += A[i * n + j] * b[j];
		}

		x[i] = acc;
	}
}

static void matrixT_vector(int m, int n, double* A, double* b, double* x) {

	for (int i=0;i<n;i++) {

		double acc = 0;
		for (int j=0;j<m;j++) {
			acc += A[i + j * n] * b[j];
		}

		x[i] = acc;
	}
}

static double vector_dot(int n, double* a, double* b) {

	double dot = 0;
	for (int i=0;i<n;i++) {
		dot += a[i] * b[i];
	}
	return dot;
}

static void calculate_gramian(int m, int n, double* A, double* G) {

	for (int k=0;k<n;k++) {
		for (int j=0;j<n;j++) {

			double acc = 0;
			for (int i=0;i<m;i++) {
				acc += A[i * n + k] * A[i * n + j];
			}

			G[k * n + j] = acc;
		}
	}
}

static void update(int n, int i, int a, int* features, double* Q, double* pQ, double* L,
			double* c, double* pc, double* x, double* y) {

	for (int k=0;k<i;k++) {
		int f = features[k];
		pQ[i * n + k] = Q[a * n + f];
		pQ[k * n + i] = Q[a * n + f];
	}

	pQ[i * n + i] = Q[a * n + a];
	pc[i] = c[a];
	cholesky_update(n, L, pQ, i);

	// TODO: probably don't need to update all of y
	forward_substitution(n, i + 1, L, pc, y);
	backward_substitution(n, i + 1, L, y, x);
}

static void _fselect(int m, int n, double* A, double* b, int* features, double* weights) {

	std::vector< int > remaining(n, 0);
	for (int i=0;i<n;i++) {
		remaining[i] = i;
	}

	std::vector< double > Ldata(n * n, 0);
	std::vector< double > Qdata(n * n, 0);
	std::vector< double > pQdata(n * n, 0);
	std::vector< double > cdata(n, 0);
	std::vector< double > pcdata(n, 0);
	std::vector< double > xdata(n, 0);
	std::vector< double > ydata(n, 0);
	double* L = Ldata.data();
	double* Q = Qdata.data();
	double* c = cdata.data();
	double* pQ = pQdata.data();
	double* x = xdata.data();
	double* y = ydata.data();
	double* pc = pcdata.data();

	double constant = vector_dot(m, b, b);
	calculate_gramian(m, n, A, Q);
	matrixT_vector(m, n, A, b, c);


	for (int i=0;i<n;i++) {
		int best_index = -1, best_feature = -1;
		double best_obj = INFINITY;

		for (int j=0;j<(int)remaining.size();j++) {
			int trial = remaining[j];

			update(n, i, trial, features, Q, pQ, L, c, pc, x, y);

			// calculate sum of squared residuals
			matrix_vector(n, i + 1, pQ, x, y);
			double obj = vector_dot(i + 1, x, y) - 2 * vector_dot(i + 1, pc, x) + constant;
			if (obj < best_obj) {
				best_index = j;
				best_feature = trial;
				best_obj = obj;
			}
		}

		remaining[best_index] = remaining.back();
		remaining.pop_back();
		update(n, i, best_feature, features, Q, pQ, L, c, pc, x, y);
		features[i] = best_feature;

		for (int j=0;j<i+1;j++) {
			weights[i + features[j] * n] = x[j];
		}
	}
}


#ifdef __cplusplus
extern "C" {
#endif

void fselect(int m, int n, double* A, double* b, int* features, double* weights) {
	_fselect(m, n, A, b, features, weights);
}

#ifdef __cplusplus
}
#endif

