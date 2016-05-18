
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "bingham/util.h"
#include "bingham/hypersphere.h"



// cached tessellations of S^3
#define MAX_LEVELS 20
hypersphere_tessellation_t tessellations[MAX_LEVELS];

#define PROXIMITY_QUEUE_SIZE 100
#define PROXIMITY_QUEUE_MEMORY_LIMIT 1e6



/*
 * Reproject mesh vertices onto the unit hypersphere.
 */
static void reproject_vertices(double **vertices, int nv, int dim)
{
  int i;
  for (i = 0; i < nv; i++) {
    double d = norm(vertices[i], dim);
    mult(vertices[i], vertices[i], 1/d, dim);
    //printf("reprojected vertex %d: %f -> %f\n", i, d, norm(vertices[i], dim));
  }
}


/*
 * Create the initial (low-res) mesh of S3 (in R4).
 */
static octetramesh_t *init_mesh_S3_octetra()
{
  octetramesh_t *mesh;
  safe_calloc(mesh, 1, octetramesh_t);
  octetramesh_new(mesh, 8, 16, 0, 4);

  double vertices[8][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1},
			   {-1,0,0,0}, {0,-1,0,0}, {0,0,-1,0}, {0,0,0,-1}};

  int tetrahedra[16][4] = {{0,1,2,3}, {0,1,2,7}, {0,1,6,3}, {0,5,2,3},
			   {0,1,6,7}, {0,5,2,7}, {0,5,6,3}, {0,5,6,7},
			   {4,1,2,3}, {4,1,2,7}, {4,1,6,3}, {4,5,2,3},
			   {4,1,6,7}, {4,5,2,7}, {4,5,6,3}, {4,5,6,7}};

  memcpy(mesh->vertices[0], vertices, 8*4*sizeof(double));
  memcpy(mesh->tetrahedra[0], tetrahedra, 16*4*sizeof(int));

  return mesh;
}


/*
 * Create a tesselation of the 3-sphere (in R4) at a given level (# of subdivisions)
 */
static octetramesh_t *build_octetra(int level)
{
  int i;
  octetramesh_t *mesh = init_mesh_S3_octetra();
  octetramesh_t tmp;

  for (i = 0; i < level; i++) {
    octetramesh_subdivide(&tmp, mesh);
    octetramesh_free(mesh);
    free(mesh);
    mesh = octetramesh_clone(&tmp);
    octetramesh_free(&tmp);
  }

  reproject_vertices(mesh->vertices, mesh->nv, mesh->d);

  return mesh;
}


/*
 * Fill in the fields of a hypersphere_tessellation from an octetramesh.
 */
static void octetramesh_to_tessellation(hypersphere_tessellation_t *T, octetramesh_t *mesh)
{
  // get tetramesh
  T->tetramesh = octetramesh_to_tetramesh(mesh);

  // dimensions
  T->d = 4;
  T->n = T->tetramesh->nt;
  int n = T->n, d = T->d;

  // compute cell centroids and volumes
  T->centroids = new_matrix2(n, d);
  safe_calloc(T->volumes, n, double);
  tetramesh_centroids(T->centroids, T->volumes, T->tetramesh);
  reproject_vertices(T->centroids, n, d);

  /**********************************
  // compute cell radii
  safe_calloc(T->radii, n, double);
  for (i = 0; i < n; i++) {
    //double V = T->volumes[i];
    //double a = cbrt(12*V/sqrt(2));  // edge length (for a regular tetrahedron)
    double *v0 = T->tetramesh->vertices[ T->tetramesh->tetrahedra[i][0] ];
    double *v1 = T->tetramesh->vertices[ T->tetramesh->tetrahedra[i][1] ];
    double a = dist(v0, v1, d);
    //printf("a = %f, %f\n", a, dist(v0,v1,d));

    T->radii[i] = sqrt(3.0/8.0)*a;
  }

  // compute proximity queues (memory permitting)
  if (n < PROXIMITY_QUEUE_SIZE || n*PROXIMITY_QUEUE_SIZE > PROXIMITY_QUEUE_MEMORY_LIMIT)
    T->proximity_queue_size = 0;
  else {
    T->proximity_queue_size = PROXIMITY_QUEUE_SIZE;
    int p = T->proximity_queue_size;
    double min_dists[n];

    T->proximity_queues = new_matrix2i(n, p);
    T->proximity_queue_dists = new_matrix2(n, p);

    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++)
	min_dists[j] = MAX(0, dist(T->centroids[i], T->centroids[j], d) - T->radii[i] - T->radii[j]);
      min_dists[i] = DBL_MAX;

      //for (j = 0; j < n; j++)
      //  printf("min_dists[%d] = %f, centroid dist = %f, r[i]=%f, r[j]=%f\n",
      //       j, min_dists[j], dist(T->centroids[i], T->centroids[j], d), T->radii[i], T->radii[j]);

      //if (1)
      //  return;  //dbug


      mink(min_dists, T->proximity_queues[i], n, p);
      for (j = 0; j < p; j++)
	T->proximity_queue_dists[i][j] = min_dists[T->proximity_queues[i][j]];
    }
  }

  //printf("n = %d, proximity queue size = %d\n", n, T->proximity_queue_size);

  //for (i = 0; i < T->proximity_queue_size; i++)
  //  printf("proximity_queue_dists[0][%d] = %f\n", i, T->proximity_queue_dists[0][i]);

  *****************************/
}


/*
 * Pre-cache some tessellations of hyperspheres.
 */
void hypersphere_init()
{
  double t0 = get_time_ms();

  int i;
  const int levels = 5; //7;

  memset(tessellations, 0, MAX_LEVELS*sizeof(hypersphere_tessellation_t));

  for (i = 0; i < levels; i++) {
    octetramesh_t *mesh = build_octetra(i);
    octetramesh_to_tessellation(&tessellations[i], mesh);
  }

  fprintf(stderr, "Initialized %d hypersphere tessellations (up to %d cells) in %.0f ms\n",
	  levels, tessellations[levels-1].n, get_time_ms() - t0);
}


/*
 * Returns a tesselation of the 3-sphere (in R4) with at least n cells.
 */
hypersphere_tessellation_t *tessellate_S3(int n)
{
  int i;
  for (i = 0; i < MAX_LEVELS; i++) {
    if (tessellations[i].tetramesh == NULL) {
      octetramesh_t *mesh = build_octetra(i);
      octetramesh_to_tessellation(&tessellations[i], mesh);
    }
    if (tessellations[i].tetramesh->nt >= n)
      return &tessellations[i];
  }

  return &tessellations[MAX_LEVELS-1];
}







//-------------------- DEPRECATED ------------------//


/*
 * Create the initial (low-res) mesh of S3 (in R4).
 *
static tetramesh_t *init_mesh_S3_tetra()
{
  tetramesh_t *T;
  safe_calloc(T, 1, tetramesh_t);
  tetramesh_new(T, 8, 16, 4);

  double vertices[8][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1},
			   {-1,0,0,0}, {0,-1,0,0}, {0,0,-1,0}, {0,0,0,-1}};

  int tetrahedra[16][4] = {{0,1,2,3}, {0,1,2,7}, {0,1,6,3}, {0,5,2,3},
			   {0,1,6,7}, {0,5,2,7}, {0,5,6,3}, {0,5,6,7},
			   {4,1,2,3}, {4,1,2,7}, {4,1,6,3}, {4,5,2,3},
			   {4,1,6,7}, {4,5,2,7}, {4,5,6,3}, {4,5,6,7}};

  memcpy(T->vertices[0], vertices, 8*4*sizeof(double));
  memcpy(T->tetrahedra[0], tetrahedra, 16*4*sizeof(int));

  return T;
}
*/

/*
 * Create a tesselation of the 3-sphere (in R4) with at least n cells.
 *
tetramesh_t *hypersphere_tessellation_tetra(int n)
{
  int i;
  tetramesh_t *T = init_mesh_S3_tetra();
  tetramesh_t tmp;

  // div = k s.t. nt*8^k > n  ==> 8^k > n/nt
  int div = (int) ceil(log(n/(double)T->nt) / log(8));  // number of times to subdivide
  div = MAX(div, 0);

  for (i = 0; i < div; i++) {

    tetramesh_subdivide(&tmp, T);
    printf("\nSubdivision %d...\n", i+1);

    tetramesh_free(T);
    free(T);
    T = tetramesh_clone(&tmp);

    tetramesh_free(&tmp);
  }

  reproject_vertices(T->vertices, T->nv, T->d);
  printf("\nProjection...\n");
  tetramesh_print_stats(tetramesh_stats(T));

  return T;
}
*/

/*
 * Create a tesselation of the 3-sphere (in R4) with at least n cells.
 *
octetramesh_t *hypersphere_tessellation_octetra(int n)
{
  int i;
  octetramesh_t *mesh = init_mesh_S3_octetra();
  octetramesh_t tmp;

  for (i = 0; i < 100000; i++) {

    if (mesh->nt + mesh->no >= n)
      break;

    octetramesh_subdivide(&tmp, mesh);
    printf("\nSubdivision %d...\n", i+1);

    octetramesh_free(mesh);
    free(mesh);
    mesh = octetramesh_clone(&tmp);

    octetramesh_free(&tmp);
  }

  reproject_vertices(mesh->vertices, mesh->nv, mesh->d);
  printf("\nProjection...\n");
  //octetramesh_print_stats(octetramesh_stats(mesh));

  return mesh;
}
*/

/* Multi-resolution tessellation of the 3-sphere based on approximating
 * a scalar function f:S3->R to a given resolution.
 *
octetramesh_t *hypersphere_tessellation_octetra_mres(double(*f)(double *, void *), void *fdata, double resolution)
{
  int i;
  octetramesh_t *mesh = init_mesh_S3_octetra();
  octetramesh_t tmp;

  for (i = 0; i < 100000; i++) {

    octetramesh_subdivide_mres(&tmp, mesh, f, fdata, resolution);

    int nv = mesh->nv;
    int nv2 = tmp.nv;

    if (nv == nv2)  // no subdivision performed
      break;

    printf("\nSubdivision %d;  nv: %d -> %d\n", i+1, nv, nv2);

    octetramesh_free(mesh);
    free(mesh);
    mesh = octetramesh_clone(&tmp);
    octetramesh_free(&tmp);
  }

  reproject_vertices(mesh->vertices, mesh->nv, mesh->d);
  printf("\nProjection...\n");
  octetramesh_print_stats(octetramesh_stats(mesh));

  return mesh;
}
*/

