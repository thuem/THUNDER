#include "cuthunder.h"

#include "easylogging++.h"

#include <cstdlib>
#include <vector>

#include <cuda_profiler_api.h>

using std::vector;

const int dimsize = 200;
const int num_images = 1500;

inline void generate_volume(double **volume, int ndim)
{
    *volume = (double*)malloc(ndim * ndim * (ndim / 2 + 1)
                              * sizeof(double) * 2);
}

void generate_images(vector<cuthunder::Complex*>& images,
                     int ndim,
                     int length)
{
    images.clear();

    double *data;
    for (int i = 0; i < length; i++) {
        data = (double*)malloc(ndim * (ndim / 2 + 1)
                               * sizeof(double) * 2);
        images.push_back(reinterpret_cast<cuthunder::Complex*>(data));
    }
}

void free_images(vector<cuthunder::Complex*>& images)
{
    vector<cuthunder::Complex*>::iterator itr = images.begin();
    for (; itr < images.end(); itr++) {
        free(*itr);
    }

    images.clear();
}

inline void generate_matrix(double **matdat, int length)
{
    *matdat = (double*)malloc(length * 9 * sizeof(double));
}

void generate_tabfunc(double **tabdat,
                      double begin,
                      double end,
                      double step,
                      int *tabsize)
{
    int counter = 1;
    for (double t = begin; t < end; t += step)
        counter++;

    *tabdat = (double*)malloc(counter * sizeof(double));

    *tabsize = counter;
}

_INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[])
{
    vector<cuthunder::Complex*> images;
    double *volume, *rotmat, *symmat, *tabfunc;
    double begin = 0.0, end = 1.0, step = 0.00001;
    int tabsize;

    generate_volume(&volume, dimsize);

    generate_images(images, dimsize, num_images);

    generate_matrix(&rotmat, num_images);

    generate_matrix(&symmat, 20);

    generate_tabfunc(&tabfunc, begin, end, step, &tabsize);

    LOG(INFO) << "Data prepare done!";

    cudaProfilerStart();
/*
    cuthunder::InsertF(images,
                       reinterpret_cast<cuthunder::Complex*>(volume),
                       dimsize,
                       rotmat, num_images,
                       symmat, 20,
                       tabfunc, begin, end, step, tabsize,
                       0);
*/
    cudaProfilerStop();

    LOG(INFO) << "Insert done!";

    free_images(images); free(volume); free(tabfunc);
    free(rotmat); free(symmat);

    LOG(INFO) << "Free done!";

	return 0;
}
