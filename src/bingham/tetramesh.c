
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "bingham/util.h"
#include "bingham/tetramesh.h"


/** Tools for subdividing and smoothing a tetrahedral mesh **/


meshgraph_t *tetramesh_meshgraph(tetramesh_t *mesh)
{
  int i, i0, i1, i2, i3, nv = mesh->nv, nt = mesh->nt;

  meshgraph_t *g = meshgraph_new(nv, 10);

  // add the vertices
  for (i = 0; i < nv; i++)
    g->vertices[i] = i;
  g->nv = nv;

  // add faces (and edges)
  for (i = 0; i < nt; i++) {
    i0 = mesh->tetrahedra[i][0];
    i1 = mesh->tetrahedra[i][1];
    i2 = mesh->tetrahedra[i][2];
    i3 = mesh->tetrahedra[i][3];
    meshgraph_add_face(g, i0, i1, i2);
    meshgraph_add_face(g, i0, i1, i3);
    meshgraph_add_face(g, i0, i2, i3);
    meshgraph_add_face(g, i1, i2, i3);
  }

  return g;
}


/*
 * Convert a tetramesh to a set of centroids and volumes.
 */
void tetramesh_centroids(double **centroids, double *volumes, tetramesh_t *mesh)
{
  int i, d = mesh->d;
  double x[d], *v0, *v1, *v2, *v3;

  for (i = 0; i < mesh->nt; i++) {   // for each tetrahedron, compute the center and volume
    memset(x, 0, d*sizeof(double));
    v0 = mesh->vertices[ mesh->tetrahedra[i][0] ];
    v1 = mesh->vertices[ mesh->tetrahedra[i][1] ];
    v2 = mesh->vertices[ mesh->tetrahedra[i][2] ];
    v3 = mesh->vertices[ mesh->tetrahedra[i][3] ];

    add(x, x, v0, d);
    add(x, x, v1, d);
    add(x, x, v2, d);
    add(x, x, v3, d);
    mult(centroids[i], x, 1/4.0, d);

    volumes[i] = tetrahedron_volume(v0, v1, v2, v3, d);
  }
}


/*
 * Build the graph of a mesh.
 */
graph_t *tetramesh_graph(tetramesh_t *mesh)
{
  int i, i0, i1, i2, i3;
  graph_t *g;
  safe_malloc(g, 1, graph_t);

  g->nv = mesh->nv;
  safe_calloc(g->vertices, g->nv, vertex_t);

  for (i = 0; i < g->nv; i++)
    g->vertices[i].index = i;

  // build neighbor lists
  for (i = 0; i < mesh->nt; i++) {
    i0 = mesh->tetrahedra[i][0];
    i1 = mesh->tetrahedra[i][1];
    i2 = mesh->tetrahedra[i][2];
    i3 = mesh->tetrahedra[i][3];

    vertex_t *v0 = &g->vertices[i0];
    vertex_t *v1 = &g->vertices[i1];
    vertex_t *v2 = &g->vertices[i2];
    vertex_t *v3 = &g->vertices[i3];

    if (!ilist_contains(v0->neighbors, i1))  v0->neighbors = ilist_add(v0->neighbors, i1);
    if (!ilist_contains(v0->neighbors, i2))  v0->neighbors = ilist_add(v0->neighbors, i2);
    if (!ilist_contains(v0->neighbors, i3))  v0->neighbors = ilist_add(v0->neighbors, i3);

    if (!ilist_contains(v1->neighbors, i0))  v1->neighbors = ilist_add(v1->neighbors, i0);
    if (!ilist_contains(v1->neighbors, i2))  v1->neighbors = ilist_add(v1->neighbors, i2);
    if (!ilist_contains(v1->neighbors, i3))  v1->neighbors = ilist_add(v1->neighbors, i3);

    if (!ilist_contains(v2->neighbors, i0))  v2->neighbors = ilist_add(v2->neighbors, i0);
    if (!ilist_contains(v2->neighbors, i1))  v2->neighbors = ilist_add(v2->neighbors, i1);
    if (!ilist_contains(v2->neighbors, i3))  v2->neighbors = ilist_add(v2->neighbors, i3);

    if (!ilist_contains(v3->neighbors, i0))  v3->neighbors = ilist_add(v3->neighbors, i0);
    if (!ilist_contains(v3->neighbors, i1))  v3->neighbors = ilist_add(v3->neighbors, i1);
    if (!ilist_contains(v3->neighbors, i2))  v3->neighbors = ilist_add(v3->neighbors, i2);
  }

  // get the total number of edges:  ne = sum(degree_i) / 2
  g->ne = 0;
  for (i = 0; i < g->nv; i++)
    g->ne += g->vertices[i].neighbors->len;
  g->ne /= 2;

  // allocate space for the vertex edge lists
  for (i = 0; i < g->nv; i++)
    safe_calloc(g->vertices[i].edges, g->vertices[i].neighbors->len, int);

  // get the edges, and build the vertex edge lists
  safe_calloc(g->edges, g->ne, edge_t);
  int cnt = 0;
  for (i = 0; i < g->nv; i++) {

    ilist_t *tmp;
    int ti = 0;  // index of neighbor list
    for (tmp = g->vertices[i].neighbors; tmp; tmp = tmp->next, ti++) {

      if (tmp->x >= i) {
	int j = tmp->x;

	// add the edge to g->edges
	g->edges[cnt].i = i;
	g->edges[cnt].j = j;
	//g->edges[cnt].len = dist(mesh->vertices[i], mesh->vertices[j], mesh->d);

	// add the edge index to the vertex edge lists of vertices i and j
	g->vertices[i].edges[ti] = cnt;
	int tj = ilist_find(g->vertices[j].neighbors, i);
	g->vertices[j].edges[tj] = cnt;

	cnt++;
      }
    }
  }

  return g;
}


/*
 * Count the number of distinct edges in the mesh.
 *
 *  ~~~~~  TODO: Use a graph traversal algorithm! ~~~~~
 *
int tetramesh_edge_count(tetramesh_t *mesh)
{
  int i, j, i0, i1, i2, i3;
  int nv = mesh->nv, nt = mesh->nt, cnt = 0, E[nv][nv];

  // E = 0
  for (i = 0; i < nv; i++)
    for (j = 0; j < nv; j++)
      E[i][j] = 0;

  // count the edges
  for (i = 0; i < nt; i++) {
    i0 = mesh->tetrahedra[i][0];
    i1 = mesh->tetrahedra[i][1];
    i2 = mesh->tetrahedra[i][2];
    i3 = mesh->tetrahedra[i][3];

    if (E[i0][i1] == 0) {
      E[i0][i1] = E[i1][i0] = 1;
      cnt++;
    }
    if (E[i0][i2] == 0) {
      E[i0][i2] = E[i2][i0] = 1;
      cnt++;
    }
    if (E[i0][i3] == 0) {
      E[i0][i3] = E[i3][i0] = 1;
      cnt++;
    }
    if (E[i1][i2] == 0) {
      E[i1][i2] = E[i2][i1] = 1;
      cnt++;
    }
    if (E[i1][i3] == 0) {
      E[i1][i3] = E[i3][i1] = 1;
      cnt++;
    }
    if (E[i2][i3] == 0) {
      E[i2][i3] = E[i3][i2] = 1;
      cnt++;
    }
  }

  return cnt;
}
*****/

/*
 * Count the number of distinct faces in the mesh.
 *
 *  ~~~~~  TODO: Use a graph traversal algorithm!  ~~~~~
 *
int tetramesh_face_count(tetramesh_t *mesh)
{
  int i, j, k, i0, i1, i2, i3;
  int nv = mesh->nv, nt = mesh->nt, cnt = 0, F[nv][nv][nv];

  // F = 0
  for (i = 0; i < nv; i++)
    for (j = 0; j < nv; j++)
      for (k = 0; k < nv; k++)
	F[i][j][k] = 0;

  // count the edges
  for (i = 0; i < nt; i++) {
    i0 = mesh->tetrahedra[i][0];
    i1 = mesh->tetrahedra[i][1];
    i2 = mesh->tetrahedra[i][2];
    i3 = mesh->tetrahedra[i][3];

    if (F[i0][i1][i2] == 0) {
      F[i0][i1][i2] = F[i0][i2][i1] = F[i1][i0][i2] = F[i1][i2][i0] = F[i2][i0][i1] = F[i2][i1][i0] = 1;
      cnt++;
    }
    if (F[i0][i1][i3] == 0) {
      F[i0][i1][i3] = F[i0][i3][i1] = F[i1][i0][i3] = F[i1][i3][i0] = F[i3][i0][i1] = F[i3][i1][i0] = 1;
      cnt++;
    }
    if (F[i0][i2][i3] == 0) {
      F[i0][i3][i2] = F[i0][i2][i3] = F[i3][i0][i2] = F[i3][i2][i0] = F[i2][i0][i3] = F[i2][i3][i0] = 1;
      cnt++;
    }
    if (F[i1][i2][i3] == 0) {
      F[i3][i1][i2] = F[i3][i2][i1] = F[i1][i3][i2] = F[i1][i2][i3] = F[i2][i3][i1] = F[i2][i1][i3] = 1;
      cnt++;
    }
  }

  return cnt;
}
*****/


/*
 * struct to hold a midpoint map from edges (p1,p2) -> q12
 *
typedef struct {
  int num_edges;
  double **points;
  int **map;
} midpoint_map_t;
*****/


/*
 * Free the contents of a midpoint map.
 *
void free_midpoint_map(midpoint_map_t *m)
{
  free_matrix2(m->points);
  free_matrix2i(m->map);
}
*******/


/*
 * Compute the sparse matrix of midpoints along each edge in the mesh.
 * @param midpoint_map 
 *
void tetramesh_midpoints(midpoint_map_t *midpoint_map, tetramesh_t *mesh)
{
  int i, j, i0, i1, i2, i3, cnt;
  int nv = mesh->nv, nt = mesh->nt, d = mesh->d;

  int num_edges = tetramesh_edge_count(mesh);
  double **points = new_matrix2(num_edges, d);
  int **Q = new_matrix2i(nv, nv);

  // Q = -1
  for (i = 0; i < nv; i++)
    for (j = 0; j < nv; j++)
      Q[i][j] = -1;

  midpoint_map->num_edges = num_edges;
  midpoint_map->points = points;
  midpoint_map->map = Q;

  // compute midpoints
  cnt = 0;
  for (i = 0; i < nt; i++) {
    i0 = mesh->tetrahedra[i][0];
    i1 = mesh->tetrahedra[i][1];
    i2 = mesh->tetrahedra[i][2];
    i3 = mesh->tetrahedra[i][3];

    if (Q[i0][i1] < 0) {
      Q[i0][i1] = Q[i1][i0] = cnt;
      avg(points[cnt], mesh->vertices[i0], mesh->vertices[i1], d);
      cnt++;
    }
    if (Q[i0][i2] < 0) {
      Q[i0][i2] = Q[i2][i0] = cnt;
      avg(points[cnt], mesh->vertices[i0], mesh->vertices[i2], d);
      cnt++;
    }
    if (Q[i0][i3] < 0) {
      Q[i0][i3] = Q[i3][i0] = cnt;
      avg(points[cnt], mesh->vertices[i0], mesh->vertices[i3], d);
      cnt++;
    }
    if (Q[i1][i2] < 0) {
      Q[i1][i2] = Q[i2][i1] = cnt;
      avg(points[cnt], mesh->vertices[i1], mesh->vertices[i2], d);
      cnt++;
    }
    if (Q[i1][i3] < 0) {
      Q[i1][i3] = Q[i3][i1] = cnt;
      avg(points[cnt], mesh->vertices[i1], mesh->vertices[i3], d);
      cnt++;
    }
    if (Q[i2][i3] < 0) {
      Q[i2][i3] = Q[i3][i2] = cnt;
      avg(points[cnt], mesh->vertices[i2], mesh->vertices[i3], d);
      cnt++;
    }
  }
}
*****/


/*
 * Compute the midpoints of all the edges in a mesh.
 */
void tetramesh_midpoints(double **dst, tetramesh_t *mesh, graph_t *graph)
{
  int i, j, e;
  for (e = 0; e < graph->ne; e++) {   // for each edge, compute the midpoint
    i = graph->edges[e].i;
    j = graph->edges[e].j;
    avg(dst[e], mesh->vertices[i], mesh->vertices[j], mesh->d);
  }
}


/*
 * Subdivide each tetrahedron in a mesh into 8 smaller tetrahedra.
 */
void tetramesh_subdivide(tetramesh_t *dst, tetramesh_t *src)
{
  int i;
  int p1, p2, p3, p4;
  int q12, q13, q14, q23, q24, q34;

  int nv = src->nv;
  int nt = src->nt;
  int d = src->d;

  // compute the midpoints of each edge in the mesh
  //midpoint_map_t M;
  //tetramesh_midpoints(&M, src);

  graph_t *graph = tetramesh_graph(src);

  int nv2 = nv + graph->ne;  //M.num_edges;  // old vertices plus the midpoints
  int nt2 = 8*nt;

  // allocate space for the new mesh and copy old vertices plus midpoints into dst
  tetramesh_new(dst, nv2, nt2, d);
  memcpy(dst->vertices[0], src->vertices[0], nv*d*sizeof(double));
  //memcpy(dst->vertices[0] + nv*d, M.points[0], (nv2-nv)*d*sizeof(double));
  tetramesh_midpoints(dst->vertices + nv, src, graph);

  for (i = 0; i < nt; i++) {    // for each tetrahedron in the original mesh

    // original point indices
    p1 = src->tetrahedra[i][0];
    p2 = src->tetrahedra[i][1];
    p3 = src->tetrahedra[i][2];
    p4 = src->tetrahedra[i][3];

    // new point indices
    q12 = nv + graph->vertices[p1].edges[ ilist_find(graph->vertices[p1].neighbors, p2) ];  //M.map[p1][p2];
    q13 = nv + graph->vertices[p1].edges[ ilist_find(graph->vertices[p1].neighbors, p3) ];  //M.map[p1][p3];
    q14 = nv + graph->vertices[p1].edges[ ilist_find(graph->vertices[p1].neighbors, p4) ];  //M.map[p1][p4];
    q23 = nv + graph->vertices[p2].edges[ ilist_find(graph->vertices[p2].neighbors, p3) ];  //M.map[p2][p3];
    q24 = nv + graph->vertices[p2].edges[ ilist_find(graph->vertices[p2].neighbors, p4) ];  //M.map[p2][p4];
    q34 = nv + graph->vertices[p3].edges[ ilist_find(graph->vertices[p3].neighbors, p4) ];  //M.map[p3][p4];

    int *t0 = dst->tetrahedra[8*i];
    int *t1 = dst->tetrahedra[8*i+1];
    int *t2 = dst->tetrahedra[8*i+2];
    int *t3 = dst->tetrahedra[8*i+3];
    int *t4 = dst->tetrahedra[8*i+4];
    int *t5 = dst->tetrahedra[8*i+5];
    int *t6 = dst->tetrahedra[8*i+6];
    int *t7 = dst->tetrahedra[8*i+7];

    t0[0] = p1;  t0[1] = q12;  t0[2] = q13;  t0[3] = q14;
    t1[0] = p2;  t1[1] = q12;  t1[2] = q23;  t1[3] = q24;
    t2[0] = p3;  t2[1] = q13;  t2[2] = q23;  t2[3] = q34;
    t3[0] = p4;  t3[1] = q14;  t3[2] = q24;  t3[3] = q34;
    t4[0] = q13;  t4[1] = q12;  t4[2] = q23;  t4[3] = q34;
    t5[0] = q13;  t5[1] = q12;  t5[2] = q14;  t5[3] = q34;
    t6[0] = q24;  t6[1] = q12;  t6[2] = q23;  t6[3] = q34;
    t7[0] = q24;  t7[1] = q12;  t7[2] = q14;  t7[3] = q34;
  }

  graph_free(graph);

  //free_midpoint_map(&M);
}


/*
 * Copy the contents of one mesh into another mesh.
 */
void tetramesh_copy(tetramesh_t *dst, tetramesh_t *src)
{
  memcpy(dst->vertices[0], src->vertices[0], src->nv*src->d*sizeof(double));  // copy raw vertices
  memcpy(dst->tetrahedra[0], src->tetrahedra[0], 4*src->nt*sizeof(int));      // copy raw tetrahedra
}


/*
 * Copy the vertices of one mesh into another mesh.
 */
void tetramesh_copy_vertices(tetramesh_t *dst, tetramesh_t *src)
{
  memcpy(dst->vertices[0], src->vertices[0], src->nv*src->d*sizeof(double));  // copy raw vertices
}


/*
 * Copy the tetrahedra of one mesh into another mesh.
 */
void tetramesh_copy_tetrahedra(tetramesh_t *dst, tetramesh_t *src)
{
  memcpy(dst->tetrahedra[0], src->tetrahedra[0], 4*src->nt*sizeof(int));      // copy raw tetrahedra
}


/*
 * Clone a mesh.
 */
tetramesh_t *tetramesh_clone(tetramesh_t *src)
{
  tetramesh_t *dst;
  safe_calloc(dst, 1, tetramesh_t);
  tetramesh_new(dst, src->nv, src->nt, src->d);
  tetramesh_copy(dst, src);

  return dst;
}


/*
 * Create (allocate) the contents of a mesh.
 */
void tetramesh_new(tetramesh_t *mesh, int nv, int nt, int d)
{
  mesh->nv = nv;
  mesh->nt = nt;
  mesh->d = d;

  mesh->vertices = new_matrix2(nv, d);
  mesh->tetrahedra = new_matrix2i(nt, 4);
}



/*
 * Free the contents of a tetrahedral mesh.
 */
void tetramesh_free(tetramesh_t *mesh)
{
  if (mesh->nv != 0) {
    //free(mesh->vertices[0]);
    //free(mesh->vertices);
    //free(mesh->tetrahedra[0]);  // tetra_raw
    //free(mesh->tetrahedra);

    free_matrix2(mesh->vertices);
    free_matrix2i(mesh->tetrahedra);
  }
}


/*
 *Save a colored tetrahedral mesh to PLY file.
 */
void tetramesh_save_PLY_colors(tetramesh_t *mesh, meshgraph_t *graph, char *filename, int *colors)
{
  FILE *f = fopen(filename, "w");

  int i, j, i0, i1, i2, i3;

  int num_faces = graph->nf;

  fprintf(f, "ply\n");
  fprintf(f, "format ascii 1.0\n");
  fprintf(f, "comment tetramesh model\n");  
  fprintf(f, "element vertex %d\n", mesh->nv);
  fprintf(f, "property float x\n");
  fprintf(f, "property float y\n");
  fprintf(f, "property float z\n");
  fprintf(f, "element face %d\n", num_faces);
  fprintf(f, "property list uchar int vertex_indices\n");
  if (colors) {
    fprintf(f, "property uchar red\n");
    fprintf(f, "property uchar green\n");
    fprintf(f, "property uchar blue\n");
  }
  fprintf(f, "end_header\n");

  for (i = 0; i < mesh->nv; i++) {
    for (j = 0; j < 3 /*mesh->d*/; j++)
      fprintf(f, "%f ", mesh->vertices[i][j]);
    fprintf(f, "\n");
  }

  int *face_colors = NULL;
  if (colors) {
    // determine what color each face should be
    safe_calloc(face_colors, num_faces, int);
    for (i = 0; i < mesh->nt; i++) {
      i0 = mesh->tetrahedra[i][0];
      i1 = mesh->tetrahedra[i][1];
      i2 = mesh->tetrahedra[i][2];
      i3 = mesh->tetrahedra[i][3];

      int face = meshgraph_find_face(graph, i0, i1, i2);
      if (colors[i] > face_colors[face])
	face_colors[face] = colors[i];

      face = meshgraph_find_face(graph, i0, i1, i3);
      if (colors[i] > face_colors[face])
	face_colors[face] = colors[i];

      face = meshgraph_find_face(graph, i0, i2, i3);
      if (colors[i] > face_colors[face])
	face_colors[face] = colors[i];

      face = meshgraph_find_face(graph, i1, i2, i3);
      if (colors[i] > face_colors[face])
	face_colors[face] = colors[i];
    }
  }

  // ~~~ TODO: Use a graph traversal algorithm to avoid double-counting! ~~~
  for (i = 0; i < num_faces; i++) {
    face_t face = graph->faces[i];
    if (colors) {
      color_t color = colormap[ face_colors[i] ];
      fprintf(f, "3 %d %d %d %d %d %d\n", face.i, face.j, face.k, color.r, color.g, color.b);
    }
    else
      fprintf(f, "3 %d %d %d\n", face.i, face.j, face.k);
  }

  /*
  for (i = 0; i < mesh->nt; i++) {
    i0 = mesh->tetrahedra[i][0];
    i1 = mesh->tetrahedra[i][1];
    i2 = mesh->tetrahedra[i][2];
    i3 = mesh->tetrahedra[i][3];
    if (colors) {
      fprintf(f, "3 %d %d %d %d %d %d\n", i0, i1, i2, colors[i].r, colors[i].g, colors[i].b);
      fprintf(f, "3 %d %d %d %d %d %d\n", i0, i1, i3, colors[i].r, colors[i].g, colors[i].b);
      fprintf(f, "3 %d %d %d %d %d %d\n", i0, i2, i3, colors[i].r, colors[i].g, colors[i].b);
      fprintf(f, "3 %d %d %d %d %d %d\n", i1, i2, i3, colors[i].r, colors[i].g, colors[i].b);
    }
    else {
      fprintf(f, "3 %d %d %d\n", i0, i1, i2);
      fprintf(f, "3 %d %d %d\n", i0, i1, i3);
      fprintf(f, "3 %d %d %d\n", i0, i2, i3);
      fprintf(f, "3 %d %d %d\n", i1, i2, i3);
    }
  }
  */

  fclose(f);
}

/*
 *Save a tetrahedral mesh to PLY file.
 */
void tetramesh_save_PLY(tetramesh_t *mesh, meshgraph_t *graph, char *filename)
{
  tetramesh_save_PLY_colors(mesh, graph, filename, 0);
}


/*
 * Compute statistics about tetrahedral areas, edge lengths, etc.
 */
tetramesh_stats_t tetramesh_stats(tetramesh_t *T)
{
  int i, j, e, nt = T->nt, d = T->d;
  tetramesh_stats_t stats;

  graph_t *graph = tetramesh_graph(T);
  stats.num_edges = graph->ne;

  stats.num_vertices = T->nv;
  stats.num_tetrahedra = T->nt;

  stats.min_edge_len = DBL_MAX;
  stats.max_edge_len = 0;
  stats.avg_edge_len = 0;
  stats.std_edge_len = 0;
  stats.min_skewness = DBL_MAX;
  stats.max_skewness = 0;
  stats.avg_skewness = 0;
  stats.std_skewness = 0;
  stats.min_volume = DBL_MAX;
  stats.max_volume = 0;
  stats.avg_volume = 0;
  stats.std_volume = 0;

  for (i = 0; i < nt; i++) {      // for each tetrahedron

    int i0 = T->tetrahedra[i][0];
    int i1 = T->tetrahedra[i][1];
    int i2 = T->tetrahedra[i][2];
    int i3 = T->tetrahedra[i][3];

    double *p0 = T->vertices[i0];
    double *p1 = T->vertices[i1];
    double *p2 = T->vertices[i2];
    double *p3 = T->vertices[i3];

    double d01 = dist(p0, p1, d);
    double d02 = dist(p0, p2, d);
    double d03 = dist(p0, p3, d);
    double d12 = dist(p1, p2, d);
    double d13 = dist(p1, p3, d);
    double d23 = dist(p2, p3, d);

    double d_edge[6] = {d01, d02, d03, d12, d13, d23};
    double dmax = arr_max(d_edge, 6);
    double dmin = arr_min(d_edge, 6);

    double skewness = dmax/dmin;
    if (skewness < stats.min_skewness)
      stats.min_skewness = skewness;
    if (skewness > stats.max_skewness)
      stats.max_skewness = skewness;
    stats.avg_skewness += skewness;

    double volume = tetrahedron_volume(p0, p1, p2, p3, d);
    if (volume < stats.min_volume)
      stats.min_volume = volume;
    if (volume > stats.max_volume)
      stats.max_volume = volume;
    stats.avg_volume += volume;
  }
  stats.avg_skewness /= (double)nt;
  stats.avg_volume /= (double)nt;

  for (i = 0; i < nt; i++) {      // for each tetrahedron

    int i0 = T->tetrahedra[i][0];
    int i1 = T->tetrahedra[i][1];
    int i2 = T->tetrahedra[i][2];
    int i3 = T->tetrahedra[i][3];

    double *p0 = T->vertices[i0];
    double *p1 = T->vertices[i1];
    double *p2 = T->vertices[i2];
    double *p3 = T->vertices[i3];

    double d01 = dist(p0, p1, d);
    double d02 = dist(p0, p2, d);
    double d03 = dist(p0, p3, d);
    double d12 = dist(p1, p2, d);
    double d13 = dist(p1, p3, d);
    double d23 = dist(p2, p3, d);

    double d_edge[6] = {d01, d02, d03, d12, d13, d23};
    double dmax = arr_max(d_edge, 6);
    double dmin = arr_min(d_edge, 6);

    double skewness = dmax/dmin;
    double ds = stats.avg_skewness - skewness;
    stats.std_skewness += ds*ds;

    double volume = tetrahedron_volume(p0, p1, p2, p3, d);
    double dv = stats.avg_volume - volume;
    stats.std_volume += dv*dv;
  }
  stats.std_skewness = sqrt(stats.std_skewness/(double)nt);
  stats.std_volume = sqrt(stats.std_volume/(double)nt);

  for (e = 0; e < graph->ne; e++) {
    i = graph->edges[e].i;
    j = graph->edges[e].j;
    double edge_len = dist(T->vertices[i], T->vertices[j], d);
    if (edge_len < stats.min_edge_len)
      stats.min_edge_len = edge_len;
    if (edge_len > stats.max_edge_len)
      stats.max_edge_len = edge_len;
    stats.avg_edge_len += edge_len;
  }
  stats.avg_edge_len /= (double)stats.num_edges;

  for (e = 0; e < graph->ne; e++) {
    i = graph->edges[e].i;
    j = graph->edges[e].j;
    double edge_len = dist(T->vertices[i], T->vertices[j], d);
    double de = stats.avg_edge_len - edge_len;
    stats.std_edge_len += de*de;
  }
  stats.std_edge_len = sqrt(stats.std_edge_len/(double)stats.num_edges);

  graph_free(graph);

  return stats;
}

/*
 * Print the stats of a mesh.
 */
void tetramesh_print_stats(tetramesh_stats_t stats)
{
  printf("tetramesh stats {\n");
  printf("  nv = %d, ne = %d, nt = %d\n", stats.num_vertices, stats.num_edges, stats.num_tetrahedra);

  printf("  edge_len = [%f, %f], avg: %f, std: %f\n",
	 stats.min_edge_len, stats.max_edge_len, stats.avg_edge_len, stats.std_edge_len);

  printf("  skewness = [%f, %f], avg: %f, std: %f\n",
	 stats.min_skewness, stats.max_skewness, stats.avg_skewness, stats.std_skewness);

  printf("  volume = [%f, %f], avg: %f, std: %f\n",
	 stats.min_volume, stats.max_volume, stats.avg_volume, stats.std_volume);

  printf("}\n");
}











/**********  DEPRECATED  ***********



static void adjust_edge_4d(tetramesh_t *mesh, graph_t *graph, int i, int j)
{
  int iter = 100;
  double thresh = .0001;
  
  int d = mesh->d;
  double *u = mesh->vertices[i];
  double *v = mesh->vertices[j];

  //double rx[4] = {.1*frand(), .1*frand(), .1*frand(), .1*frand()};
  //double ry[4] = {.1*frand(), .1*frand(), .1*frand(), .1*frand()};
  //add(u, u, rx, d);
  //add(v, v, ry, d);
  
  double a[4] = {0,0,0,0};
  double b[4] = {0,0,0,0};
  ilist_t *tmp;
  for (tmp = graph->vertices[i].neighbors; tmp; tmp = tmp->next)
    if (tmp->x != j)
      add(a, a, mesh->vertices[tmp->x], d);
  for (tmp = graph->vertices[j].neighbors; tmp; tmp = tmp->next)
    if (tmp->x != i)
      add(b, b, mesh->vertices[tmp->x], d);

  int k;
  for (k = 0; k < iter; k++) {
    double F[] = {(v[1]+a[1])*u[0] - (v[0]+a[0])*u[1],
		  (v[2]+a[2])*u[0] - (v[0]+a[0])*u[2],
		  (v[3]+a[3])*u[0] - (v[0]+a[0])*u[3],
		  (u[1]+b[1])*v[0] - (u[0]+b[0])*v[1],
		  (u[2]+b[2])*v[0] - (u[0]+b[0])*v[2],
		  (u[3]+b[3])*v[0] - (u[0]+b[0])*v[3],
		  norm(u, d) - 1,
		  norm(v, d) - 1};

    //printf("  u = (%f, %f, %f, %f)\n", u[0], u[1], u[2], u[3]);
    //printf("  v = (%f, %f, %f, %f)\n", v[0], v[1], v[2], v[3]);
    //printf("  a = (%f, %f, %f, %f)\n", a[0], a[1], a[2], a[3]);
    //printf("  b = (%f, %f, %f, %f)\n", b[0], b[1], b[2], b[3]);

    //printf("  k = %d, norm(F) = %f\n", k, norm(F,8));

    if (norm(F, 8) < thresh)
      break;

    double J[] = {v[1]+a[1],  -v[0]-a[0],           0,           0,       -u[1],        u[0],           0,           0,
		  v[2]+a[2],           0,  -v[0]-a[0],           0,       -u[2],           0,        u[0],           0,
		  v[3]+a[3],           0,           0,  -v[0]-a[0],       -u[3],           0,           0,        u[0],
		  -v[1],            v[0],           0,           0,   u[1]+b[1],  -u[0]-b[0],           0,           0,
		  -v[2],               0,        v[0],           0,   u[2]+b[2],           0,  -u[0]-b[0],           0,
		  -v[3],               0,           0,        v[0],   u[3]+b[3],           0,           0,  -u[0]-b[0],
		  2*u[0],         2*u[1],      2*u[2],      2*u[3],           0,           0,           0,           0,
		  0,                   0,           0,           0,      2*v[0],      2*v[1],      2*v[2],      2*v[3]};

    double duv[8];
    mult(F, F, -1, d);
    solve(duv, J, F, d);

    //printf("  duv = (%f, %f, %f, %f, %f, %f, %f, %f)\n", duv[0], duv[1], duv[2], duv[3], duv[4], duv[5], duv[6], duv[7]);

    add(u, u, duv, d);
    add(v, v, duv+d, d);
  }
}


void tetramesh_smooth_edges(tetramesh_t *dst, tetramesh_t *src, double w)
{
  graph_t *graph = tetramesh_graph(src);

  if (dst != src)
    tetramesh_copy(dst, src);
  
  int i;
  for (i = 0; i < graph->ne; i++)
    adjust_edge_4d(dst, graph, graph->edges[i].i, graph->edges[i].j);
}


void tetramesh_smooth2(tetramesh_t *dst, tetramesh_t *src, double w)
{
  int i, j, d = src->d;
  double p[d];
  ilist_t *v;
  graph_t *graph = tetramesh_graph(src);

  if (dst != src)
    tetramesh_copy(dst, src);

  for (i = 0; i < graph->nv; i++) {
    memset(p, 0, d * sizeof(double));                          // p = 0
    for (v = graph->vertices[i].neighbors; v; v = v->next) {
      j = v->x;
      //      printf("  dst->vertices[%d] = (%.2f, %.2f, %.2f, %.2f) --> dist(%d,%d) = %f\n",
      //	     j, dst->vertices[j][0], dst->vertices[j][1], dst->vertices[j][2], dst->vertices[j][3],
      //	     i, j, dist(dst->vertices[i], dst->vertices[j], d));
      add(p, p, dst->vertices[j], d);                          // p += dst->vertices[j]
    }
    mult(p, p, 1/norm(p, d), d);                               // p = p/norm(p)
    //    printf("p = (%.2f, %.2f, %.2f, %.2f), dst->vertices[i] = (%.2f, %.2f, %.2f, %.2f)\n",
    //	   p[0], p[1], p[2], p[3], dst->vertices[i][0], dst->vertices[i][1], dst->vertices[i][2], dst->vertices[i][3]);
    wavg(dst->vertices[i], p, dst->vertices[i], w, d);         // dst->vertices[i] = w*p + (1-w)*dst->vertices[i]
  }
}


// Smooth the tetrahedral mesh
void tetramesh_smooth(tetramesh_t *dst, tetramesh_t *src, double w)
{
  // Laplacian
  //static double L[4][4] = {{0, 1/3.0, 1/3.0, 1/3.0},
  //			     {1/3.0, 0, 1/3.0, 1/3.0},
  //			     {1/3.0, 1/3.0, 0, 1/3.0},
  //			     {1/3.0, 1/3.0, 1/3.0, 0}};

  int i, j, i0, i1, i2, i3;
  int nv = src->nv;
  int nt = src->nt;
  int d = src->d;
  tetramesh_t tmp;
  int cnt[nv];

  tetramesh_new(&tmp, nv, nt, d);
  memset(cnt, 0, nv*sizeof(int));

  for (i = 0; i < nt; i++) {      // for each tetrahedron

    i0 = src->tetrahedra[i][0];
    i1 = src->tetrahedra[i][1];
    i2 = src->tetrahedra[i][2];
    i3 = src->tetrahedra[i][3];

    double *p0 = src->vertices[i0];
    double *p1 = src->vertices[i1];
    double *p2 = src->vertices[i2];
    double *p3 = src->vertices[i3];

    double m0[d], m1[d], m2[d], m3[d];

    avg3(m0, p1, p2, p3, d);
    avg3(m1, p0, p2, p3, d);
    avg3(m2, p0, p1, p3, d);
    avg3(m3, p0, p1, p2, d);

    wavg(m0, m0, p0, w, d);
    wavg(m1, m1, p1, w, d);
    wavg(m2, m2, p2, w, d);
    wavg(m3, m3, p3, w, d);

    add(tmp.vertices[i0], tmp.vertices[i0], m0, d);
    add(tmp.vertices[i1], tmp.vertices[i1], m1, d);
    add(tmp.vertices[i2], tmp.vertices[i2], m2, d);
    add(tmp.vertices[i3], tmp.vertices[i3], m3, d);

    cnt[i0]++;
    cnt[i1]++;
    cnt[i2]++;
    cnt[i3]++;
  }

  for (i = 0; i < nv; i++)
    for (j = 0; j < d; j++)
      tmp.vertices[i][j] /= (double) cnt[i];

  tetramesh_copy_vertices(dst, &tmp);
  tetramesh_copy_tetrahedra(dst, src);

  tetramesh_free(&tmp);
}

******************************************************/


