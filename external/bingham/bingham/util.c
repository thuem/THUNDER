#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "bingham/util.h"
//#include <lapacke.h>
//#undef I  // fuck C99!

const color_t colormap[256] =
  {{0, 0, 131},
   {0, 0, 135},
   {0, 0, 139},
   {0, 0, 143},
   {0, 0, 147},
   {0, 0, 151},
   {0, 0, 155},
   {0, 0, 159},
   {0, 0, 163},
   {0, 0, 167},
   {0, 0, 171},
   {0, 0, 175},
   {0, 0, 179},
   {0, 0, 183},
   {0, 0, 187},
   {0, 0, 191},
   {0, 0, 195},
   {0, 0, 199},
   {0, 0, 203},
   {0, 0, 207},
   {0, 0, 211},
   {0, 0, 215},
   {0, 0, 219},
   {0, 0, 223},
   {0, 0, 227},
   {0, 0, 231},
   {0, 0, 235},
   {0, 0, 239},
   {0, 0, 243},
   {0, 0, 247},
   {0, 0, 251},
   {0, 0, 255},
   {0, 4, 255},
   {0, 8, 255},
   {0, 12, 255},
   {0, 16, 255},
   {0, 20, 255},
   {0, 24, 255},
   {0, 28, 255},
   {0, 32, 255},
   {0, 36, 255},
   {0, 40, 255},
   {0, 44, 255},
   {0, 48, 255},
   {0, 52, 255},
   {0, 56, 255},
   {0, 60, 255},
   {0, 64, 255},
   {0, 68, 255},
   {0, 72, 255},
   {0, 76, 255},
   {0, 80, 255},
   {0, 84, 255},
   {0, 88, 255},
   {0, 92, 255},
   {0, 96, 255},
   {0, 100, 255},
   {0, 104, 255},
   {0, 108, 255},
   {0, 112, 255},
   {0, 116, 255},
   {0, 120, 255},
   {0, 124, 255},
   {0, 128, 255},
   {0, 131, 255},
   {0, 135, 255},
   {0, 139, 255},
   {0, 143, 255},
   {0, 147, 255},
   {0, 151, 255},
   {0, 155, 255},
   {0, 159, 255},
   {0, 163, 255},
   {0, 167, 255},
   {0, 171, 255},
   {0, 175, 255},
   {0, 179, 255},
   {0, 183, 255},
   {0, 187, 255},
   {0, 191, 255},
   {0, 195, 255},
   {0, 199, 255},
   {0, 203, 255},
   {0, 207, 255},
   {0, 211, 255},
   {0, 215, 255},
   {0, 219, 255},
   {0, 223, 255},
   {0, 227, 255},
   {0, 231, 255},
   {0, 235, 255},
   {0, 239, 255},
   {0, 243, 255},
   {0, 247, 255},
   {0, 251, 255},
   {0, 255, 255},
   {4, 255, 251},
   {8, 255, 247},
   {12, 255, 243},
   {16, 255, 239},
   {20, 255, 235},
   {24, 255, 231},
   {28, 255, 227},
   {32, 255, 223},
   {36, 255, 219},
   {40, 255, 215},
   {44, 255, 211},
   {48, 255, 207},
   {52, 255, 203},
   {56, 255, 199},
   {60, 255, 195},
   {64, 255, 191},
   {68, 255, 187},
   {72, 255, 183},
   {76, 255, 179},
   {80, 255, 175},
   {84, 255, 171},
   {88, 255, 167},
   {92, 255, 163},
   {96, 255, 159},
   {100, 255, 155},
   {104, 255, 151},
   {108, 255, 147},
   {112, 255, 143},
   {116, 255, 139},
   {120, 255, 135},
   {124, 255, 131},
   {128, 255, 128},
   {131, 255, 124},
   {135, 255, 120},
   {139, 255, 116},
   {143, 255, 112},
   {147, 255, 108},
   {151, 255, 104},
   {155, 255, 100},
   {159, 255, 96},
   {163, 255, 92},
   {167, 255, 88},
   {171, 255, 84},
   {175, 255, 80},
   {179, 255, 76},
   {183, 255, 72},
   {187, 255, 68},
   {191, 255, 64},
   {195, 255, 60},
   {199, 255, 56},
   {203, 255, 52},
   {207, 255, 48},
   {211, 255, 44},
   {215, 255, 40},
   {219, 255, 36},
   {223, 255, 32},
   {227, 255, 28},
   {231, 255, 24},
   {235, 255, 20},
   {239, 255, 16},
   {243, 255, 12},
   {247, 255, 8},
   {251, 255, 4},
   {255, 255, 0},
   {255, 251, 0},
   {255, 247, 0},
   {255, 243, 0},
   {255, 239, 0},
   {255, 235, 0},
   {255, 231, 0},
   {255, 227, 0},
   {255, 223, 0},
   {255, 219, 0},
   {255, 215, 0},
   {255, 211, 0},
   {255, 207, 0},
   {255, 203, 0},
   {255, 199, 0},
   {255, 195, 0},
   {255, 191, 0},
   {255, 187, 0},
   {255, 183, 0},
   {255, 179, 0},
   {255, 175, 0},
   {255, 171, 0},
   {255, 167, 0},
   {255, 163, 0},
   {255, 159, 0},
   {255, 155, 0},
   {255, 151, 0},
   {255, 147, 0},
   {255, 143, 0},
   {255, 139, 0},
   {255, 135, 0},
   {255, 131, 0},
   {255, 128, 0},
   {255, 124, 0},
   {255, 120, 0},
   {255, 116, 0},
   {255, 112, 0},
   {255, 108, 0},
   {255, 104, 0},
   {255, 100, 0},
   {255, 96, 0},
   {255, 92, 0},
   {255, 88, 0},
   {255, 84, 0},
   {255, 80, 0},
   {255, 76, 0},
   {255, 72, 0},
   {255, 68, 0},
   {255, 64, 0},
   {255, 60, 0},
   {255, 56, 0},
   {255, 52, 0},
   {255, 48, 0},
   {255, 44, 0},
   {255, 40, 0},
   {255, 36, 0},
   {255, 32, 0},
   {255, 28, 0},
   {255, 24, 0},
   {255, 20, 0},
   {255, 16, 0},
   {255, 12, 0},
   {255, 8, 0},
   {255, 4, 0},
   {255, 0, 0},
   {251, 0, 0},
   {247, 0, 0},
   {243, 0, 0},
   {239, 0, 0},
   {235, 0, 0},
   {231, 0, 0},
   {227, 0, 0},
   {223, 0, 0},
   {219, 0, 0},
   {215, 0, 0},
   {211, 0, 0},
   {207, 0, 0},
   {203, 0, 0},
   {199, 0, 0},
   {195, 0, 0},
   {191, 0, 0},
   {187, 0, 0},
   {183, 0, 0},
   {179, 0, 0},
   {175, 0, 0},
   {171, 0, 0},
   {167, 0, 0},
   {163, 0, 0},
   {159, 0, 0},
   {155, 0, 0},
   {151, 0, 0},
   {147, 0, 0},
   {143, 0, 0},
   {139, 0, 0},
   {135, 0, 0},
   {131, 0, 0},
   {128, 0, 0}};


double get_time_ms()
{
  struct timeval tv;
  struct timezone tz;

  gettimeofday(&tv, &tz);

  return 1000.*tv.tv_sec + tv.tv_usec/1000.;
}


// returns a pointer to the nth word (starting from 0) in string s
char *sword(const char *s, const char *delim, int n)
{
  if (s == NULL)
    return NULL;

  s += strspn(s, delim);  // skip over initial delimeters

  int i;
  for (i = 0; i < n; i++) {
    s += strcspn(s, delim);  // skip over word
    s += strspn(s, delim);  // skip over delimeters
  }

  return (char *)s;
}


// splits a string into k words
char **split(const char *s, const char *delim, int *k)
{
  const char *sbuf = s + strspn(s, delim);  // skip over initial whitespace
  s = sbuf;

  // determine the number of words
  int num_words = 0;
  while (*s != '\0') {
    s = sword(s, delim, 1);
    num_words++;
  }

  // fill in the words
  int i;
  s = sbuf;
  char **words;
  safe_calloc(words, num_words, char *);
  for (i = 0; i < num_words; i++) {
    int slen = strcspn(s, delim);  // add "\n" ?
    safe_calloc(words[i], slen+1, char);  // +1 to null-terminate the string
    strncpy(words[i], s, slen);
    s = sword(s, delim, 1);
  }

  *k = num_words;
  return words;
}


// compare the first word of s1 with the first word of s2
int wordcmp(const char *s1, const char *s2, const char *delim)
{
  int n1 = strcspn(s1, delim);
  int n2 = strcspn(s2, delim);

  if (n1 < n2)
    return -1;
  else if (n1 > n2)
    return 1;

  return strncmp(s1, s2, n1);
}


// replace a word in a string array
void replace_word(char **words, int num_words, const char *from, const char *to)
{
  int i;
  for (i = 0; i < num_words; i++) {
    if (!strcmp(words[i], from)) {
      safe_realloc(words[i], strlen(to)+1, char);
      strcpy(words[i], to);
    }
  }
}


// computes the log factorial of x
double lfact(int x)
{
  static double logf[MAXFACT];
  static int first = 1;
  int i;

  if (first) {
    first = 0;
    logf[0] = 0;
    for (i = 1; i < MAXFACT; i++)
      logf[i] = log(i) + logf[i-1];
  }

  return logf[x];
}


// computes the factorial of x
double fact(int x)
{
  return exp(lfact(x));
}


// computes the surface area of a unit sphere with dimension d
double surface_area_sphere(int d)
{
  switch(d) {
  case 0:
    return 2;
  case 1:
    return 2*M_PI;
  case 2:
    return 4*M_PI;
  case 3:
    return 2*M_PI*M_PI;
  }

  return (2*M_PI/((double)d-1))*surface_area_sphere(d-2);
}


// logical not of a binary array
void vnot(int y[], int x[], int n)
{
  int i;
  for (i = 0; i < n; i++)
    y[i] = !x[i];
}


// count the non-zero elements of x
int count(int x[], int n)
{
  int i;
  int cnt = 0;
  for (i = 0; i < n; i++)
    if (x[i] != 0)
      cnt++;

  return cnt;
}


// returns a dense array of the indices of x's non-zero elements
int find(int *k, int x[], int n)
{
  int i;
  int cnt = 0;
  for (i = 0; i < n; i++)
    if (x[i] != 0)
      k[cnt++] = i;
  return cnt;
}


// returns a sparse array of the indices of x's non-zero elements
int findinv(int *k, int x[], int n)
{
  int i;
  int cnt = 0;
  for (i = 0; i < n; i++)
    if (x[i] != 0)
      k[i] = cnt++;
  return cnt;
}


// computes a dense array of the indices of x==a
int findeq(int *k, int x[], int a, int n)
{
  int i;
  int cnt = 0;
  if (k != NULL) {
    for (i = 0; i < n; i++) {
      if (x[i] == a)
	k[cnt++] = i;
    }
  }
  else
    for (i = 0; i < n; i++)
      if (x[i] == a)
	cnt++;
  return cnt;
}


// computes the sum of x's elements
double sum(double x[], int n)
{
  int i;
  double y = 0;
  for (i = 0; i < n; i++)
    y += x[i];
  return y;
}

// computes the product of x's elements
double prod(double x[], int n)
{
  int i;
  double y = 1;
  for (i = 0; i < n; i++)
    y *= x[i];
  return y;
}


// computes the max of x
double arr_max(double x[], int n)
{
  int i;

  double y = x[0];
  for (i = 1; i < n; i++)
    if (x[i] > y)
      y = x[i];

  return y;
}

// computes the max of x
int arr_max_i(int x[], int n)
{
  int i;

  int y = x[0];
  for (i = 1; i < n; i++)
    if (x[i] > y)
      y = x[i];

  return y;
}

// computes the masked max of x
double arr_max_masked(double x[], int mask[], int n)
{
  int i;

  for (i = 0; i < n; i++)
    if (mask[i])
      break;
  if (i==n)
    return NAN;

  double y = x[i++];
  for (; i < n; i++)
    if (mask[i] && (x[i] > y))
      y = x[i];

  return y;
}

// computes the masked max of x
float arr_maxf_masked(float x[], int mask[], int n)
{
  int i;

  for (i = 0; i < n; i++)
    if (mask[i])
      break;
  if (i==n)
    return NAN;

  float y = x[i++];
  for (; i < n; i++)
    if (mask[i] && (x[i] > y))
      y = x[i];

  return y;
}

// computes the min of x
double arr_min(double x[], int n)
{
  int i;

  double y = x[0];
  for (i = 1; i < n; i++)
    if (x[i] < y)
      y = x[i];

  return y;
}

// computes the min of x
int arr_min_i(int x[], int n)
{
  int i;

  int y = x[0];
  for (i = 1; i < n; i++)
    if (x[i] < y)
      y = x[i];

  return y;
}

// computes the masked min of x
double arr_min_masked(double x[], int mask[], int n)
{
  int i;

  for (i = 0; i < n; i++)
    if (mask[i] != 0)
      break;
  if (i==n)
    return NAN;

  double y = x[i++];
  for (; i < n; i++)
    if (mask[i] && (x[i] < y))
      y = x[i];

  return y;
}

// computes the masked min of x
float arr_minf_masked(float x[], int mask[], int n)
{
  int i;

  for (i = 0; i < n; i++)
    if (mask[i])
      break;
  if (i==n)
    return NAN;

  float y = x[i++];
  for (; i < n; i++)
    if (mask[i] && (x[i] < y))
      y = x[i];

  return y;
}

// returns the index of the max of x
int find_max(double x[], int n)
{
  int i;
  int idx = 0;
  for (i = 1; i < n; i++)
    if (x[i] > x[idx])
      idx = i;
  return idx;
}

// returns the index of the min of x
int find_min(double x[], int n)
{
  int i;
  int idx = 0;
  for (i = 1; i < n; i++)
    if (x[i] < x[idx])
      idx = i;
  return idx;
}

// returns the index of the max of x
int find_imax(int x[], int n)
{
  int i;
  int idx = 0;
  for (i = 1; i < n; i++)
    if (x[i] > x[idx])
      idx = i;
  return idx;
}

// returns the index of the min of x
int find_imin(int x[], int n)
{
  int i;
  int idx = 0;
  for (i = 1; i < n; i++)
    if (x[i] < x[idx])
      idx = i;
  return idx;
}

// computes the sum of x's elements
int isum(int x[], int n)
{
  int i;
  int y = 0;
  for (i = 0; i < n; i++)
    y += x[i];
  return y;
}

// computes the max of x
int imax(int x[], int n)
{
  int i;

  int y = x[0];
  for (i = 1; i < n; i++)
    if (x[i] > y)
      y = x[i];

  return y;
}


// computes the min of x
int imin(int x[], int n)
{
  int i;

  int y = x[0];
  for (i = 1; i < n; i++)
    if (x[i] < y)
      y = x[i];

  return y;
}


// computes the norm of x
double norm(double x[], int n)
{
  double d = 0.0;
  int i;

  for (i = 0; i < n; i++)
    d += x[i]*x[i];

  return sqrt(d);
}

// computes the norm of x-y
double dist(double x[], double y[], int n)
{
  double d = 0.0;
  int i;

  for (i = 0; i < n; i++)
    d += (x[i]-y[i])*(x[i]-y[i]);

  return sqrt(d);
}


// computes the norm^2 of x-y
double dist2(double x[], double y[], int n)
{
  double d = 0.0;
  int i;

  for (i = 0; i < n; i++)
    d += (x[i]-y[i])*(x[i]-y[i]);

  return d;
}


// computes the dot product of z and y
double dot(double x[], double y[], int n)
{
  int i;
  double z = 0.0;
  for (i = 0; i < n; i++)
    z += x[i]*y[i];
  return z;
}


//computes the cross product of x and y
void cross(double z[3], double x[3], double y[3])
{
  z[0] = x[1]*y[2] - x[2]*y[1];
  z[1] = x[2]*y[0] - x[0]*y[2];
  z[2] = x[0]*y[1] - x[1]*y[0];
}

void cross4d(double w[4], double x[4], double y[4], double z[4]) {
  double **V = new_matrix2(4, 3);
  int i;
  for (i = 0; i < 4; ++i) {
    V[i][0] = x[i];
    V[i][1] = y[1];
    V[i][2] = z[i];
  }
  double **W = new_matrix2(3, 3);
  int indices[4][3] = {{2, 3, 4}, {1, 3, 4}, {1, 2, 4}, {1, 2, 3}};
  int cut_axis = 0;
  double dmax = 0;
  for (i = 0; i < 4; ++i) {
    reorder_rows(W, V, indices[i], 3, 3);
    double tmp = fabs(det(W, 3));
    if (dmax < tmp) {
      dmax = tmp;
      cut_axis = i;
    }    
  }
  double *c0 = V[cut_axis];
  int uncut_axis[3];
  for (i = 0; i < cut_axis; ++i) {
    uncut_axis[i] = i;
    w[i] = 0;
  }
  for (i = cut_axis + 1; i < 4; ++i) {
    uncut_axis[i-1] = i;
    w[i] = 0;
  }
  w[cut_axis] = 1;
  reorder_rows(W, V, uncut_axis, 3, 3);
  inv(W, W, 3);
  mult(c0, c0, -1, 3);
  
  double tmp[3];
  matrix_vec_mult(tmp, W, c0, 3, 3); 
  for (i = 0; i < 3; ++i) {
    w[uncut_axis[i]] = tmp[i];
  }
  free_matrix2(V);
  free_matrix2(W);
}

// adds two vectors, z = x+y
void add(double z[], double x[], double y[], int n)
{
  int i;
  for (i = 0; i < n; i++)
    z[i] = x[i] + y[i];
}


// subtracts two vectors, z = x-y
void sub(double z[], double x[], double y[], int n)
{
  int i;
  for (i = 0; i < n; i++)
    z[i] = x[i] - y[i];
}


// multiplies a vector by a scalar, y = c*x
void mult(double y[], double x[], double c, int n)
{
  int i;
  for (i = 0; i < n; i++)
    y[i] = c*x[i];
}

// computes the cumulative sum of x
void cumsum(double y[], double x[], int n)
{
  int i;
  double c = 0;
  for (i = 0; i < n; i++) {
    c += x[i];
    y[i] = c;
  }
}

// takes absolute value element-wise
void vec_func(double y[], double x[], double n, double (*f)(double)) {
  int i;
  for (i = 0; i < n; ++i) {
    y[i] = (*f)(x[i]);
  }
}

// sets y = x/norm(x)
void normalize(double y[], double x[], int n)
{
  double d = norm(x, n);
  int i;
  for (i = 0; i < n; i++)
    y[i] = x[i]/d;
}


// sets y = x/sum(x)
void normalize_pmf(double y[], double x[], int n)
{
  double d = sum(x, n);
  int i;
  for (i = 0; i < n; i++)
    y[i] = x[i]/d;
}


// multiplies two vectors, z = x.*y
void vmult(double z[], double x[], double y[], int n)
{
  int i;
  for (i = 0; i < n; i++)
    z[i] = x[i]*y[i];
}


// averages two vectors, z = (x+y)/2
void avg(double z[], double x[], double y[], int n)
{
  add(z, x, y, n);
  mult(z, z, .5, n);
}


// averages two vectors, z = w*x+(1-w)*y
void wavg(double z[], double x[], double y[], double w, int n)
{
  int i;
  for (i = 0; i < n; i++)
    z[i] = w*x[i] + (1-w)*y[i];
}


// averages three vectors, y = (x1+x2+x3)/3
void avg3(double y[], double x1[], double x2[], double x3[], int n)
{
  add(y, x1, x2, n);
  add(y, y, x3, n);
  mult(y, y, 1/3.0, n);
}


// calculate the projection of x onto y
void proj(double z[], double x[], double y[], int n)
{
  double u[n];  // y's unit vector
  double d = norm(y, n);
  mult(u, y, 1/d, n);
  mult(z, u, dot(x,u,n), n);
}


// binary search to find i s.t. A[i-1] <= x < A[i]
int binary_search(double x, double *A, int n)
{
  int i0 = 0;
  int i1 = n-1;
  int i;

  while (i0 <= i1) {
    i = (i0 + i1) / 2;
    if (x > A[i])
      i0 = i + 1;
    else if (i > 0 && x < A[i-1])
      i1 = i-1;
    else
      break;
  }

  if (i0 <= i1)
    return i;

  return n-1;
}

void plane_from_3points(double *coeffs, double *p0, double *p1, double *p2)
{
    double diff1[3], diff2[3];
    sub(diff1, p1, p0, 3);
    sub(diff2, p2, p0, 3);
    double normal[3];
    cross(normal, diff1, diff2);
    normalize(normal, normal, 3);
    double flip = normal[2] > 0.0 ? -1.0 : 1.0; // flip normal towards camera
    
    coeffs[0] = normal[0] * flip;
    coeffs[1] = normal[1] * flip;
    coeffs[2] = normal[2] * flip;
    coeffs[3] = -dot(normal, p0, 3) * flip;
}

// quaternion multiplication:  z = x*y
void quaternion_mult(double z[4], double x[4], double y[4])
{
  double a = x[0];
  double b = x[1];
  double c = x[2];
  double d = x[3];
  double y0 = y[0];
  double y1 = y[1];
  double y2 = y[2];
  double y3 = y[3];

  z[0] = a*y0 - b*y1 - c*y2 - d*y3;
  z[1] = b*y0 + a*y1 - d*y2 + c*y3;
  z[2] = c*y0 + d*y1 + a*y2 - b*y3;
  z[3] = d*y0 - c*y1 + b*y2 + a*y3;
}


// invert a quaternion
void quaternion_inverse(double q_inv[4], double q[4])
{
  q_inv[0] = q[0];
  q_inv[1] = -q[1];
  q_inv[2] = -q[2];
  q_inv[3] = -q[3];
}


// quaternion exponentiation (q2 = q^a)
void quaternion_pow(double q2[4], double q[4], double a)
{
  double u[3];  // axis of rotation
  normalize(u, &q[1], 3);
  double w = MIN(MAX(q[0], -1.0), 1.0);  // for numerical stability
  double theta2 = acos(w);  // theta / 2.0
  double s = sin(a*theta2);
  q2[0] = cos(a*theta2);
  mult(&q2[1], u, s, 3);
}


// quaternion interpolation (slerp)
void quaternion_interpolation(double q[4], double q0[4], double q1[4], double t)
{
  double q0_inv[4], q01[4];
  quaternion_inverse(q0_inv, q0);
  quaternion_mult(q01, q1, q0_inv);
  quaternion_pow(q01, q01, t);
  quaternion_mult(q, q01, q0);
}


// convert a rotation matrix to a unit quaternion
void rotation_matrix_to_quaternion(double *q, double **R)
{
  double S;
  double tr = R[0][0] + R[1][1] + R[2][2];
  if (tr > 0) {
    S = sqrt(tr+1.0) * 2;  // S=4*qw
    q[0] = 0.25 * S;
    q[1] = (R[2][1] - R[1][2]) / S;
    q[2] = (R[0][2] - R[2][0]) / S;
    q[3] = (R[1][0] - R[0][1]) / S;
  }
  else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
    S = sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2;  // S=4*qx 
    q[0] = (R[2][1] - R[1][2]) / S;
    q[1] = 0.25 * S;
    q[2] = (R[0][1] + R[1][0]) / S; 
    q[3] = (R[0][2] + R[2][0]) / S; 
  }
  else if (R[1][1] > R[2][2]) {
    S = sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2;  // S=4*qy
    q[0] = (R[0][2] - R[2][0]) / S;
    q[1] = (R[0][1] + R[1][0]) / S; 
    q[2] = 0.25 * S;
    q[3] = (R[1][2] + R[2][1]) / S; 
  }
  else {
    S = sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2;  // S=4*qz
    q[0] = (R[1][0] - R[0][1]) / S;
    q[1] = (R[0][2] + R[2][0]) / S;
    q[2] = (R[1][2] + R[2][1]) / S;
    q[3] = 0.25 * S;
  }

  normalize(q, q, 4);
}


// convert a unit quaternion to a rotation matrix
void quaternion_to_rotation_matrix(double **R, double *q)
{
  double a = q[0];
  double b = q[1];
  double c = q[2];
  double d = q[3];

  R[0][0] = a*a + b*b - c*c - d*d;
  R[0][1] = 2*b*c - 2*a*d;
  R[0][2] = 2*b*d + 2*a*c;
  R[1][0] = 2*b*c + 2*a*d;
  R[1][1] = a*a - b*b + c*c - d*d;
  R[1][2] = 2*c*d - 2*a*b;
  R[2][0] = 2*b*d - 2*a*c;
  R[2][1] = 2*c*d + 2*a*b;
  R[2][2] = a*a - b*b - c*c + d*d;
}

int find_first_non_zero(double *v, int n)
{
  int i;
  for (i = 0; i < n; ++i) {
    if (v[i] != 0.)
      return i;
  }
  return -1;
}

int find_first_lt(double *x, double a, int n)
{
  int i;
  for (i = 0; i < n; ++i)
    if (x[i] < a)
      break;
  return i;
}

int find_first_gt(double *x, double a, int n)
{
  int i;
  for (i = 0; i < n; ++i)
    if (x[i] > a)
      break;
  return i;
}

/*
short *ismember(double *A, double *B, int n, int m) {
  short *C;
  safe_calloc(C, n, short);
  int i, j;
  // NOTE(sanja): this can be done in O(n log n + m) if necessary. Also, SSE2-able :)
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      if (double_is_equal(A[i], B[j])) {
	C[i] = 1;
	break;
      }
    }
  }
  return C;
}

short *ismemberi(int *A, int *B, int n, int m) {
  short *C;
  safe_calloc(C, n, short);
  int i, j;
  // NOTE(sanja): this can be done in O(n log n + m) if necessary. Also, SSE2-able :)
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      if (A[i] == B[j]) {
	C[i] = 1;
	break;
      }
    }
  }
  return C;
}
*/

// check if y contains x
int ismemberi(int x, int *y, int n)
{
  int i;
  for (i = 0; i < n; ++i)
    if (x == y[i])
      return 1;
  return 0;
}

// reverses an array of doubles (safe for x==y)
void reverse(double *y, double *x, int n)
{
  int i;
  for (i = 0; i < n/2; i++) {
    double tmp = x[i];
    y[i] = x[n-i-1];
    y[n-i-1] = tmp;
  }
}

// reverses an array of ints (safe for x==y)
void reversei(int *y, int *x, int n)
{
  int i;
  for (i = 0; i < n/2; i++) {
    int tmp = x[i];
    y[i] = x[n-i-1];
    y[n-i-1] = tmp;
  }
}

// reorder an array of doubles (safe for x==y)
void reorder(double *y, double *x, int *idx, int n)
{
  int i;
  double *y2 = y;
  if (x==y)
    safe_calloc(y2, n, double);
  for (i = 0; i < n; i++)
    y2[i] = x[idx[i]];
  if (x==y) {
    memcpy(y, y2, n*sizeof(double));
    free(y2);
  }
}

// reorder an array of ints (safe for x==y)
void reorderi(int *y, int *x, int *idx, int n)
{
  int i;
  int *y2 = y;
  if (x==y)
    safe_calloc(y2, n, int);
  for (i = 0; i < n; i++)
    y2[i] = x[idx[i]];
  if (x==y) {
    memcpy(y, y2, n*sizeof(int));
    free(y2);
  }
}

// add an element to the front of a list
ilist_t *ilist_add(ilist_t *x, int a)
{
  ilist_t *head;
  safe_malloc(head, 1, ilist_t);
  head->x = a;
  head->next = x;
  head->len = (x ? 1 + x->len : 1);

  return head;
}


// check if a list contains an element
int ilist_contains(ilist_t *x, int a)
{
  if (!x)
    return 0;

  ilist_t *tmp;
  for (tmp = x; tmp; tmp = tmp->next)
    if (tmp->x == a)
      return 1;
  return 0;
}


// find the index of an element in a list (or -1 if not found)
int ilist_find(ilist_t *x, int a)
{
  int i = 0;
  ilist_t *tmp;
  for (tmp = x; tmp; tmp = tmp->next) {
    if (tmp->x == a)
      return i;
    i++;
  }

  return -1;
}


// free a list
void ilist_free(ilist_t *x)
{
  ilist_t *tmp, *tmp2;
  tmp = x;
  while (tmp) {
    tmp2 = tmp->next;
    free(tmp);
    tmp = tmp2;
  }  
}


static void init_rand()
{
  static int first = 1;
  if (first) {
    first = 0;
    int seed = time(NULL); 
    //    seed = 1371836140;
    //int seed = 1368560954; <-- Crashes straw bowl on 3/9
    //1368457226; <--- Shows overlap on 5/3
    printf("********* seed = %d\n", seed);
    srand (seed);
  }
}


// returns a random int between 0 and n-1
int irand(int n)
{
  init_rand();

  if (n < 0)
    printf("Negative n: %d\n", n);
  return rand() % n;
}


// returns a random double in [0,1]
double frand()
{
  init_rand();

  return fabs(rand()) / (double)RAND_MAX;
}


// samples d integers from 0:n-1 uniformly without replacement
void randperm(int *x, int n, int d)
{
  init_rand();

  int i;

  if (d > n) {
    fprintf(stderr, "Error: d > n in randperm()\n");
    return;
  }
  
  // sample a random starting point
  int i0 = rand() % n;

  // use a random prime step to cycle through x
  static const int big_primes[100] = {996311, 163573, 481123, 187219, 963323, 103769, 786979, 826363, 874891, 168991, 442501, 318679, 810377, 471073, 914519, 251059, 321983, 220009, 211877, 875339, 605603, 578483, 219619, 860089, 644911, 398819, 544927, 444043, 161717, 301447, 201329, 252731, 301463, 458207, 140053, 906713, 946487, 524389, 522857, 387151, 904283, 415213, 191047, 791543, 433337, 302989, 445853, 178859, 208499, 943589, 957331, 601291, 148439, 296801, 400657, 829637, 112337, 134707, 240047, 669667, 746287, 668243, 488329, 575611, 350219, 758449, 257053, 704287, 252283, 414539, 647771, 791201, 166031, 931313, 787021, 520529, 474667, 484361, 358907, 540271, 542251, 825829, 804709, 664843, 423347, 820367, 562577, 398347, 940349, 880603, 578267, 644783, 611833, 273001, 354329, 506101, 292837, 851017, 262103, 288989};

  int step = big_primes[rand() % 100];

  int idx = i0;
  for (i = 0; i < d; i++) {
    x[i] = idx;
    idx = (idx + step) % n;
  }

  /*
  if (d > 2*sqrt(n*log(n))) {
    double r[n];
    int idx[n];
    for (i = 0; i < n; i++)
      r[i] = frand();
    sort_indices(r, idx, n);
    memcpy(x, idx, d*sizeof(int));
  }
  else {
    for (i = 0; i < d; i++) {
      while (1) {
	x[i] = rand() % n;
	for (j = 0; j < i; j++)
	  if (x[j] == x[i])
	    break;
	if (j == i)  // x[i] is unique
	  break;
      }
    }
  }
  */
}

// approximation to the inverse error function
double erfinv(double x)
{
  if (x < 0)
    return -erfinv(-x);

  double a = .147;

  double y1 = (2/(M_PI*a) + log(1-x*x)/2.0);
  double y2 = sqrt(y1*y1 - (1/a)*log(1-x*x));
  double y3 = sqrt(y2 - y1);
  
  return y3;
}


// generate a random sample from a normal distribution
double normrand(double mu, double sigma)
{
  double u = frand();
  
  return mu + sigma*sqrt(2.0)*erfinv(2*u-1);
}


// compute the pdf of a normal random variable
double normpdf(double x, double mu, double sigma)
{
  double dx = x - mu;

  return exp(-dx*dx / (2*sigma*sigma)) / (sqrt(2*M_PI) * sigma);
}


// samples from the probability mass function w with n elements
int pmfrand(double *w, int n) {

  int i;
  double r = frand();
  double wtot = 0;
  for (i = 0; i < n; i++) {
    wtot += w[i];
    if (wtot >= r)
      return i;
  }

  return 0;
}

// samples from the cumulative mass function w with n elements (much faster than pmfrand)
int cmfrand(double *w, int n)
{
  double r = frand();
  return binary_search(r, w, n);
}

// sample from a multivariate normal
void mvnrand(double *x, double *mu, double **S, int d)
{
  double z[d], **V = new_matrix2(d,d);
  eigen_symm(z,V,S,d);
  int i;
  for (i = 0; i < d; i++)
    z[i] = sqrt(z[i]);

  mvnrand_pcs(x,mu,z,V,d);

  free_matrix2(V);
}

double mvnpdf(double *x, double *mu, double **S, int d)
{
  double **S_inv = new_matrix2(d,d);
  inv(S_inv, S, d);
  
  double dx[d];
  sub(dx, x, mu, d);
  double S_inv_dx[d];
  matrix_vec_mult(S_inv_dx, S_inv, dx, d, d);
  double dm = dot(dx, S_inv_dx, d);

  double p = exp(-.5*dm) / sqrt(pow(2*M_PI, d) * det(S,d));

  free_matrix2(S_inv);

  return p;

}

/* compute a multivariate normal pdf
double mvnpdf(double *x, double *mu, double **S, int d)
{
  double z[d], **V = new_matrix2(d,d);
  eigen_symm(z,V,S,d);
  int i;
  for (i = 0; i < d; i++)
    z[i] = sqrt(z[i]);

  printf("S = [%f %f %f; %f %f %f; %f %f %f]\n", S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2]); //dbug
  printf("z = [%f, %f, %f]\n", z[0], z[1], z[2]); //dbug

  double p = mvnpdf_pcs(x,mu,z,V,d);

  free_matrix2(V);
  return p;
}
*/

// sample from a multivariate normal in principal components form
void mvnrand_pcs(double *x, double *mu, double *z, double **V, int d)
{
  int i;
  double s, v[d];

  memcpy(x, mu, d*sizeof(double));

  for (i = 0; i < d; i++) {
    s = normrand(0, z[i]);
    mult(v, V[i], s, d);  // v = s*V[i]
    add(x, x, v, d);      // x += v
  }
}


// compute a multivariate normal pdf in principal components form
double mvnpdf_pcs(double *x, double *mu, double *z, double **V, int d)
{
  int i;
  double xv, dx[d];
  sub(dx, x, mu, d);  // dx = x - mu

  double logp = -(d/2)*log(2*M_PI) - log(prod(z,d));
  for (i = 0; i < d; i++) {
    xv = dot(dx, V[i], d) / z[i];
    logp -= 0.5*xv*xv;
  }

  return exp(logp);
}


// sample from an angular central gaussian in principal components form
void acgrand_pcs(double *x, double *z, double **V, int d)
{
  int i;
  double mu[d];
  for (i = 0; i < d; i++)
    mu[i] = 0;

  mvnrand_pcs(x, mu, z, V, d);
  normalize(x, x, d);
}


// compute an angular central gaussian pdf in principal components form
double acgpdf_pcs(double *x, double *z, double **V, int d)
{
  int i;
  double p = 1 / (prod(z,d) * surface_area_sphere(d-1));
  double xv, md = 0;  // mahalanobis distance
  for (i = 0; i < d; i++) {
    xv = dot(x, V[i], d) / z[i];
    md += xv*xv;
  }
  p *= pow(md, -d/2);
  
  return p;
}


// create a new n-by-m-by-p 3d matrix of doubles
double ***new_matrix3(int n, int m, int p)
{
  if (n*m*p == 0) return NULL;
  int i;
  double **X2 = new_matrix2(n*m, p);
  double ***X;
  safe_malloc(X, n, double**);
  for (i = 0; i < n; i++)
    X[i] = X2 + m*i;

  return X;
}

// free a 3d matrix
void free_matrix3(double ***X)
{
  free_matrix2(X[0]);
  free(X);
}

// create a new n-by-m-by-p 3d matrix of floats
float ***new_matrix3f(int n, int m, int p)
{
  if (n*m*p == 0) return NULL;
  int i;
  float **X2 = new_matrix2f(n*m, p);
  float ***X;
  safe_malloc(X, n, float**);
  for (i = 0; i < n; i++)
    X[i] = X2 + m*i;

  return X;
}

// free a 3d matrix
void free_matrix3f(float ***X)
{
  free_matrix2f(X[0]);
  free(X);
}

// copy a 3d matrix of doubles: Y = X
void matrix3_copy(double ***Y, double ***X, int n, int m, int p)
{
  memcpy(Y[0][0], X[0][0], n*m*p*sizeof(double));
}

// clone a 3d matrix of doubles: Y = new(X)
double ***matrix3_clone(double ***X, int n, int m, int p)
{
  double ***Y = new_matrix3(n,m,p);
  matrix3_copy(Y, X, n, m, p);

  return Y;
}

// create a new n-by-m 2d matrix of doubles
double **new_matrix2(int n, int m)
{
  if (n*m == 0) return NULL;
  int i;
  double *raw, **X;
  safe_calloc(raw, n*m, double);
  safe_malloc(X, n, double*);

  for (i = 0; i < n; i++)
    X[i] = raw + m*i;

  return X;
}

void add_rows_matrix2(double ***X, int n, int m, int new_n)
{
  int i;
  double *raw = (*X)[0];
  safe_realloc(raw, m * new_n, double);
  safe_realloc(*X, new_n, double*);
  for (i = 0; i < new_n; i++)
    (*X)[i] = raw + m*i;
}

void add_rows_matrix2i(int ***X, int n, int m, int new_n)
{
  int i;
  int *raw = (*X)[0];
  safe_realloc(raw, m * new_n, int);
  safe_realloc(*X, new_n, int*);
  for (i = 0; i < new_n; i++)
    (*X)[i] = raw + m*i;
}


/*
void resize_matrix2(double ***X, int n, int m, int n2, int m2)
{
  if (m2 == m)
    add_rows_matrix2(X, n, m, n2);
  else {
    
  }
}
*/

// create a new n-by-m 2d matrix of floats
float **new_matrix2f(int n, int m)
{
  if (n*m == 0) return NULL;
  int i;
  float *raw, **X;
  safe_calloc(raw, n*m, float);
  safe_malloc(X, n, float*);

  for (i = 0; i < n; i++)
    X[i] = raw + m*i;

  return X;
}

// create a new n-by-m 2d matrix of ints
int **new_matrix2i(int n, int m)
{
  if (n*m == 0) return NULL;
  int i, *raw, **X;
  safe_calloc(raw, n*m, int);
  safe_malloc(X, n, int*);

  for (i = 0; i < n; i++)
    X[i] = raw + m*i;

  return X;
}

// create a new n-by-m 2d matrix of chars
char **new_matrix2c(int n, int m)
{
  if (n*m == 0) return NULL;
  int i;
  char *raw, **X;
  safe_calloc(raw, n*m, char);
  safe_malloc(X, n, char*);

  for (i = 0; i < n; i++)
    X[i] = raw + m*i;

  return X;
}

// create a new n-by-m 2d matrix of doubles
double **new_matrix2_data(int n, int m, double *data)
{
  double **X = new_matrix2(n,m);
  memcpy(X[0], data, n*m*sizeof(double));
  return X;
}

// create a new n-by-m 2d matrix of floats
float **new_matrix2f_data(int n, int m, float *data)
{
  float **X = new_matrix2f(n,m);
  memcpy(X[0], data, n*m*sizeof(float));
  return X;
}

// create a new n-by-m 2d matrix of ints
int **new_matrix2i_data(int n, int m, int *data)
{
  int **X = new_matrix2i(n,m);
  memcpy(X[0], data, n*m*sizeof(int));
  return X;
}

// create a new n-by-m 2d matrix of chars
char **new_matrix2c_data(int n, int m, char *data)
{
  char **X = new_matrix2c(n,m);
  memcpy(X[0], data, n*m*sizeof(char));
  return X;
}

/*
double **add_matrix_row(double **X, int n, int m)
{
  //printf("DANGER! Reallocating matrix rows is not tested yet!\n");
  double *raw = X[0];
  safe_realloc(raw, (n + 1) * m, double);
  safe_realloc(X, n+1, double*);
  X[n] = raw + m * n;

  return X;
}
*/

double **new_identity_matrix2(int n) {
  double **mat = new_matrix2(n, n);
  int i;
  for (i = 0; i < n; ++i)
    mat[i][i] = 1;
  return mat;
}

int **new_identity_matrix2i(int n) {
  int **mat = new_matrix2i(n, n);
  int i;
  for (i = 0; i < n; ++i)
    mat[i][i] = 1;
  return mat;
}

double **new_diag_matrix2(double *diag, int n) {
  double **mat = new_matrix2(n, n);
  int i;
  for (i = 0; i < n; ++i) {
    mat[i][i] = diag[i];
  }
  return mat;
}

int **new_diag_matrix2i(int *diag, int n) {
  int **mat = new_matrix2i(n, n);
  int i;
  for (i = 0; i < n; ++i) {
    mat[i][i] = diag[i];
  }
  return mat;
}

// free a 2d matrix of doubles
void free_matrix2(double **X)
{
  if (X == NULL) return;
  free(X[0]);
  free(X);
}

// free a 2d matrix of floats
void free_matrix2f(float **X)
{
  if (X == NULL) return;
  free(X[0]);
  free(X);
}

// free a 2d matrix of ints
void free_matrix2i(int **X)
{
  if (X == NULL) return;
  free(X[0]);
  free(X);
}

// free a 2d matrix of chars
void free_matrix2c(char **X)
{
  if (X == NULL) return;
  free(X[0]);
  free(X);
}

/*
 * Write a matrix in the following format.
 *
 * <nrows> <ncols>
 * <row 1>
 * <row 2>
 * ...
 */
void save_matrix(const char *fout, double **X, int n, int m)
{
  //fprintf(stderr, "saving matrix to %s\n", fout);

  FILE *f = fopen(fout, "w");
  int i, j;

  fprintf(f, "%d %d\n", n, m);
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++)
      fprintf(f, "%f ", X[i][j]);
    fprintf(f, "\n");
  }

  fclose(f);
}

void save_matrixi(const char *fout, int **X, int n, int m)
{
  //fprintf(stderr, "saving matrix to %s\n", fout);

  FILE *f = fopen(fout, "w");
  int i, j;

  fprintf(f, "%d %d\n", n, m);
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++)
      fprintf(f, "%d ", X[i][j]);
    fprintf(f, "\n");
  }

  fclose(f);
}

/*
 * Write a 3d matrix in the following format.
 *
 * <ntabs> <nrows> <ncols>
 * <tab 1 row 1>
 * <tab 1 row 2>
 * ...
 * <tab 2 row 1>
 * <tab 2 row 2>
 * ...
 */
void save_matrix3(const char *fout, double ***X, int n, int m, int p)
{
  //fprintf(stderr, "saving matrix3 to %s\n", fout);

  FILE *f = fopen(fout, "w");
  int i, j, k;

  fprintf(f, "%d %d %d\n", n, m, p);
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      for (k = 0; k < p; k++)
	fprintf(f, "%f ", X[i][j][k]);
      fprintf(f, "\n");
    }
  }

  fclose(f);
}

/*
 * Load a matrix in the following format.
 *
 * <nrows> <ncols>
 * <row 1>
 * <row 2>
 * ...
 */
double **load_matrix(char *fin, int *n, int *m)
{
  FILE *f = fopen(fin, "r");

  if (f == NULL) {
    fprintf(stderr, "Invalid filename: %s\n", fin);
    return NULL;
  }

  char sbuf0[128], *s = sbuf0;
  if (fgets(s, 128, f) == NULL || sscanf(s, "%d %d", n, m) < 2) {
    fprintf(stderr, "Corrupt matrix header in file %s\n", fin);
    fclose(f);
    return NULL;
  }

  double **X = new_matrix2(*n, *m);

  const int CHARS_PER_FLOAT = 20;
  char sbuf[CHARS_PER_FLOAT * (*m)];

  int i, j;
  for (i = 0; i < *n; i++) {
    s = sbuf;
    if (fgets(s, 10000, f) == NULL)
      break;
    for (j = 0; j < *m; j++) {
      if (sscanf(s, "%lf", &X[i][j]) < 1)
	break;
      s = sword(s, " \t", 1);
    }
    if (j < *m)
      break;
  }
  if (i < *n) {
    fprintf(stderr, "Corrupt matrix file '%s' at line %d\n", fin, i+2);
    fclose(f);
    free_matrix2(X);
    return NULL;
  }

  fclose(f);

  return X;
}

/*
 * Load a 3d matrix in the following format.
 *
 * <ntabs> <nrows> <ncols>
 * <tab 1 row 1>
 * <tab 1 row 2>
 * ...
 * <tab 2 row 1>
 * <tab 2 row 2>
 * ...
 */
double ***load_matrix3(char *fin, int *n, int *m, int *p)
{
  FILE *f = fopen(fin, "r");

  if (f == NULL) {
    fprintf(stderr, "Invalid filename: %s\n", fin);
    return NULL;
  }

  char sbuf0[128], *s = sbuf0;
  if (fgets(s, 128, f) == NULL || sscanf(s, "%d %d %d", n, m, p) < 3) {
    fprintf(stderr, "Corrupt matrix header in file %s\n", fin);
    fclose(f);
    return NULL;
  }

  double ***X = new_matrix3(*n, *m, *p);

  const int CHARS_PER_FLOAT = 20;
  char sbuf[CHARS_PER_FLOAT * (*p)];

  int i, j, k;
  for (i = 0; i < *n; i++) {
    for (j = 0; j < *m; j++) {
      s = sbuf;
      if (fgets(s, 10000, f) == NULL)
	break;
      for (k = 0; k < *p; k++) {
	if (sscanf(s, "%lf", &X[i][j][k]) < 1)
	  break;
	s = sword(s, " \t", 1);
      }
      if (k < *p)
	break;
    }
    if (j < *m)
      break;
  }
  if (i < *n) {
    fprintf(stderr, "Corrupt matrix file '%s' at line %d\n", fin, i*(*m)+j+2);
    fclose(f);
    free_matrix3(X);
    return NULL;
  }

  fclose(f);

  return X;
}


// calculate the area of a triangle
double triangle_area(double x[], double y[], double z[], int n)
{
  double a = dist(x, y, n);
  double b = dist(x, z, n);
  double c = dist(y, z, n);
  double s = .5*(a + b + c);

  return sqrt(s*(s-a)*(s-b)*(s-c));
}


// calculate the volume of a tetrahedron
double tetrahedron_volume(double x1[], double x2[], double x3[], double x4[], int n)
{
  double U = dist2(x1, x2, n);
  double V = dist2(x1, x3, n);
  double W = dist2(x2, x3, n);
  double u = dist2(x3, x4, n);
  double v = dist2(x2, x4, n);
  double w = dist2(x1, x4, n);

  double a = v+w-U;
  double b = w+u-V;
  double c = u+v-W;

  return sqrt( (4*u*v*w - u*a*a - v*b*b - w*c*c + a*b*c) ) / 12.0 ;
}


// calculate the volume of a tetrahedron
inline double tetrahedron_volume_old(double x[], double y[], double z[], double w[], int n)
{
  // make an orthonormal basis in the xyz plane (with x at the origin)
  double u[n], v[n], v_proj[n];
  sub(u, y, x, n);             // u = y-x
  sub(v, z, x, n);             // v = z-x
  proj(v_proj, v, u, n);       // project v onto u
  sub(v, v, v_proj, n);        // v -= v_proj
  mult(u, u, 1/norm(u,n), n);  // normalize u
  mult(v, v, 1/norm(v,n), n);  // normalize v

  // project (w-x) onto xyz plane
  double w2[n], wu[n], wv[n], w_proj[n];
  sub(w2, w, x, n);            // w2 = w-x
  proj(wu, w2, u, n);          // project w2 onto u
  proj(wv, w2, v, n);          // project w2 onto v
  add(w_proj, wu, wv, n);      // w_proj = wu + wv
  sub(w2, w2, w_proj, n);      // w2 -= w_proj

  double h = norm(w2, n);  // height
  double A = triangle_area(x, y, z, n);

  return h*A/3.0;
}


// transpose a matrix
void transpose(double **Y, double **X, int n, int m)
{
  double **X2 = X;
  if (Y == X)
    X2 = matrix_clone(X,n,m);

  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      Y[j][i] = X2[i][j];

  if (Y == X)
    free_matrix2(X2);
}


/*
int test_matrix_copy()
{
  double X_data[6] = {1,2,3,4,5,6};
  double **X = new_matrix2_data(3,2, X_data);

  //double **X = new_matrix2(2,2);
  //X[0][0] = 1;
  //X[0][1] = 2;
  //X[1][0] = 3;
  //X[1][1] = 4;

  

  
  // 1 2
  // 3 4


}
*/

// matrix copy, Y = X 
void matrix_copy(double **Y, double **X, int n, int m)
{
  memcpy(Y[0], X[0], n*m*sizeof(double));
}


// matrix clone, Y = new(X)
double **matrix_clone(double **X, int n, int m)
{
  double **Y = new_matrix2(n,m);
  matrix_copy(Y, X, n, m);

  return Y;
}


// matrix addition, Z = X+Y
void matrix_add(double **Z, double **X, double **Y, int n, int m)
{
  add(Z[0], X[0], Y[0], n*m);
}

// matrix subtraction, Z = X-Y
void matrix_sub(double **Z, double **X, double **Y, int n, int m)
{
  sub(Z[0], X[0], Y[0], n*m);
}

// matrix multiplication, Z = X*Y, where X is n-by-p and Y is p-by-m
void matrix_mult(double **Z, double **X, double **Y, int n, int p, int m)
{
  double **Z2 = (Z==X || Z==Y ? new_matrix2(n,m) : Z);
  int i, j, k;
  for (i = 0; i < n; i++) {     // row i
    for (j = 0; j < m; j++) {   // column j
      Z2[i][j] = 0;
      for (k = 0; k < p; k++)
	Z2[i][j] += X[i][k]*Y[k][j];
    }
  }
  if (Z==X || Z==Y) {
    matrix_copy(Z, Z2, n, m);
    free_matrix2(Z2);
  }
}


// matrix-vector multiplication, y = A*x
void matrix_vec_mult(double *y, double **A, double *x, int n, int m)
{
  int i;
  if (y == x) {
    double z[m];
    memcpy(z, x, m*sizeof(double));
    for (i = 0; i < n; i++)
      y[i] = dot(A[i], z, m);
  }
  else
    for (i = 0; i < n; i++)
      y[i] = dot(A[i], x, m);
}

// vector-matrix multiplication, y = x*A
void vec_matrix_mult(double *y, double *x, double **A, int n, int m)
{
  int i, j;
  if (y == x) {
    double z[n];
    memcpy(z, x, n*sizeof(double));
    for (j = 0; j < m; j++) {
      y[j] = 0;
      for (i = 0; i < n; i++)
	y[j] += z[i]*A[i][j];
    }
  }
  else {
    for (j = 0; j < m; j++) {
      y[j] = 0;
      for (i = 0; i < n; i++)
	y[j] += x[i]*A[i][j];
    }
  }
}

// matrix element-wise multiplication
void matrix_elt_mult(double **Z, double **X, double **Y, int n, int m) {
  int i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      Z[i][j] = X[i][j] * Y[i][j];
    }
  }
}

void matrix_pow(double **Y, double **X, int n, int m, double pw) {
  int i, j;
  for (i = 0; i < n; ++i)
    for (j = 0; j < m; ++j)
      Y[i][j] = pow(X[i][j], pw);
}

void matrix_sum(double y[], double **X, int n, int m) {
  int i, j;
  memset(y, 0, n * sizeof(double));
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      y[j] += X[i][j];
    }
  }
}

// outer product of x and y, Z = x'*y
void outer_prod(double **Z, double x[], double y[], int n, int m)
{
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      Z[i][j] = x[i]*y[j];
}


// row vector min
void row_min(double *y, double **X, int n, int m)
{
  int i,j;
  memcpy(y, X[0], m*sizeof(double));
  for (i = 1; i < n; i++)
    for (j = 0; j < m; j++)
      if (X[i][j] < y[j])
	y[j] = X[i][j];
}


// row vector max
void row_max(double *y, double **X, int n, int m)
{
  int i,j;
  memcpy(y, X[0], m*sizeof(double));
  for (i = 1; i < n; i++)
    for (j = 0; j < m; j++)
      if (X[i][j] > y[j])
	y[j] = X[i][j];
}


// row vector mean 
// NOTE(sanja): this is adding up columns, not rows.
void mean(double *mu, double **X, int n, int m)
{
  memset(mu, 0, m*sizeof(double));  // mu = 0

  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      mu[j] += X[i][j];

  mult(mu, mu, 1/(double)n, m);
}

void variance(double *vars, double **X, int n, int m)
{
  double mu[m];
  mean(mu, X, n, m);
  memset(vars, 0, m*sizeof(double));
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      double dx = X[i][j] - mu[j];
      vars[j] += dx*dx;
    }
  }
  mult(vars, vars, 1/(double)n, m);
}

// compute the covariance of the rows of X, given mean mu
void cov(double **S, double **X, double *mu, int n, int m)
{
  int i, j, k;

  memset(S[0], 0, m*m*sizeof(double));
  double dx[m];

  if (m == 3) {
    for (i = 0; i < n; i++) {
      dx[0] = X[i][0] - mu[0];
      dx[1] = X[i][1] - mu[1];
      dx[2] = X[i][2] - mu[2];
      S[0][0] += dx[0]*dx[0];
      S[0][1] += dx[0]*dx[1];
      S[0][2] += dx[0]*dx[2];
      S[1][0] += dx[1]*dx[0];
      S[1][1] += dx[1]*dx[1];
      S[1][2] += dx[1]*dx[2];
      S[2][0] += dx[2]*dx[0];
      S[2][1] += dx[2]*dx[1];
      S[2][2] += dx[2]*dx[2];
    }
  }
  else {
    for (i = 0; i < n; i++) {
      sub(dx, X[i], mu, m);
      for (j = 0; j < m; j++)
	for (k = 0; k < m; k++)
	  S[j][k] += dx[j]*dx[k];
    }
  }

  /*
  double **dX = matrix_clone(X, n, m);
  if (mu != NULL)
    for (i = 0; i < n; i++)
      sub(dX[i], X[i], mu, m);
  double **dXt = new_matrix2(m, n);
  transpose(dXt, dX, n, m);
  matrix_mult(S, dXt, dX, m, n, m);
  */
  mult(S[0], S[0], 1/(double)n, m*m);

  //free_matrix2(dX);
  //free_matrix2(dXt);
}


// weighted row vector mean
void wmean(double *mu, double **X, double *w, int n, int m)
{
  memset(mu, 0, m*sizeof(double));  // mu = 0

  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      mu[j] += w[i]*X[i][j];

  mult(mu, mu, 1.0/sum(w,n), m);
}


// compute the weighted covariance of the rows of X, given mean mu
void wcov(double **S, double **X, double *w, double *mu, int n, int m)
{
  int i;

  memset(S[0], 0, m*m*sizeof(double));

  double **Si = new_matrix2(m,m);
  for (i = 0; i < n; i++) {
    outer_prod(Si, X[i], X[i], m, m);
    mult(Si[0], Si[0], w[i], m*m);
    matrix_add(S, S, Si, m, m);
  }

  mult(S[0], S[0], 1.0/sum(w,n), m*m);

  if (mu != NULL) {
    double **S_mu = new_matrix2(m,m);
    outer_prod(S_mu, mu, mu, m, m);
    sub(S[0], S[0], S_mu[0], m*m);
    free_matrix2(S_mu);
  }
}


// solve the equation Ax = b, where A is a square n-by-n matrix
void solve(double *x, double **A, double *b, int n)
{
  double **A_inv = new_matrix2(n,n);
  inv(A_inv, A, n);

  int i;
  for (i = 0; i < n; i++)
    x[i] = dot(A_inv[i], b, n);

  free_matrix2(A_inv);
}


// compute the determinant of the n-by-n matrix X
double det(double **X, int n)
{
  if (n == 1)
    return X[0][0];

  else if (n == 2)
    return X[0][0]*X[1][1] - X[0][1]*X[1][0];

  else if (n == 3) {
    double a = X[0][0];
    double b = X[0][1];
    double c = X[0][2];
    double d = X[1][0];
    double e = X[1][1];
    double f = X[1][2];
    double g = X[2][0];
    double h = X[2][1];
    double i = X[2][2];
    return a*e*i - a*f*h + b*f*g - b*d*i + c*d*h - c*e*g;
  }

  else if (n == 4) {
    double a00 = X[0][0];
    double a01 = X[0][1];
    double a02 = X[0][2];
    double a03 = X[0][3];
    double a10 = X[1][0];
    double a11 = X[1][1];
    double a12 = X[1][2];
    double a13 = X[1][3];
    double a20 = X[2][0];
    double a21 = X[2][1];
    double a22 = X[2][2];
    double a23 = X[2][3];
    double a30 = X[3][0];
    double a31 = X[3][1];
    double a32 = X[3][2];
    double a33 = X[3][3];

    return a00*a11*a22*a33 - a00*a11*a23*a32 - a00*a12*a21*a33 + a00*a12*a23*a31 + a00*a13*a21*a32
      - a00*a13*a22*a31 - a01*a10*a22*a33 + a01*a10*a23*a32 + a01*a12*a20*a33 - a01*a12*a23*a30
      - a01*a13*a20*a32 + a01*a13*a22*a30 + a02*a10*a21*a33 - a02*a10*a23*a31 - a02*a11*a20*a33
      + a02*a11*a23*a30 + a02*a13*a20*a31 - a02*a13*a21*a30 - a03*a10*a21*a32 + a03*a10*a22*a31
      + a03*a11*a20*a32 - a03*a11*a22*a30 - a03*a12*a20*a31 + a03*a12*a21*a30;

  }

  else {
    fprintf(stderr, "Error: det() not supported for > 4x4 matrices\n");
    exit(1);
  }

  return 0;
}


// compute the inverse (Y) of the n-by-n matrix X
void inv(double **Y, double **X, int n)
{
  double d = det(X,n);

  if (n == 1)
    Y[0][0] = 1/d;

  else if (n == 2) {
    Y[0][0] = X[1][1] / d;
    Y[0][1] = -X[0][1] / d;
    Y[1][0] = -X[1][0] / d;
    Y[1][1] = X[0][0] / d;
  }

  else if (n == 3) {
    Y[0][0] = (X[1][1]*X[2][2] - X[1][2]*X[2][1]) / d;
    Y[0][1] = (X[0][2]*X[2][1] - X[0][1]*X[2][2]) / d;
    Y[0][2] = (X[0][1]*X[1][2] - X[0][2]*X[1][1]) / d;
    Y[1][0] = (X[1][2]*X[2][0] - X[1][0]*X[2][2]) / d;
    Y[1][1] = (X[0][0]*X[2][2] - X[0][2]*X[2][0]) / d;
    Y[1][2] = (X[0][2]*X[1][0] - X[0][0]*X[1][2]) / d;
    Y[2][0] = (X[1][0]*X[2][1] - X[1][1]*X[2][0]) / d;
    Y[2][1] = (X[0][1]*X[2][0] - X[0][0]*X[2][1]) / d;
    Y[2][2] = (X[0][0]*X[1][1] - X[0][1]*X[1][0]) / d;
  }

  else if (n == 4) {
    double a00 = X[0][0];
    double a01 = X[0][1];
    double a02 = X[0][2];
    double a03 = X[0][3];
    double a10 = X[1][0];
    double a11 = X[1][1];
    double a12 = X[1][2];
    double a13 = X[1][3];
    double a20 = X[2][0];
    double a21 = X[2][1];
    double a22 = X[2][2];
    double a23 = X[2][3];
    double a30 = X[3][0];
    double a31 = X[3][1];
    double a32 = X[3][2];
    double a33 = X[3][3];

    Y[0][0] = (a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) / d;
    Y[0][1] = (a01*a23*a32 - a01*a22*a33 + a02*a21*a33 - a02*a23*a31 - a03*a21*a32 + a03*a22*a31) / d;
    Y[0][2] = (a01*a12*a33 - a01*a13*a32 - a02*a11*a33 + a02*a13*a31 + a03*a11*a32 - a03*a12*a31) / d;
    Y[0][3] = (a01*a13*a22 - a01*a12*a23 + a02*a11*a23 - a02*a13*a21 - a03*a11*a22 + a03*a12*a21) / d;
    Y[1][0] = (a10*a23*a32 - a10*a22*a33 + a12*a20*a33 - a12*a23*a30 - a13*a20*a32 + a13*a22*a30) / d;
    Y[1][1] = (a00*a22*a33 - a00*a23*a32 - a02*a20*a33 + a02*a23*a30 + a03*a20*a32 - a03*a22*a30) / d;
    Y[1][2] = (a00*a13*a32 - a00*a12*a33 + a02*a10*a33 - a02*a13*a30 - a03*a10*a32 + a03*a12*a30) / d;
    Y[1][3] = (a00*a12*a23 - a00*a13*a22 - a02*a10*a23 + a02*a13*a20 + a03*a10*a22 - a03*a12*a20) / d;
    Y[2][0] = (a10*a21*a33 - a10*a23*a31 - a11*a20*a33 + a11*a23*a30 + a13*a20*a31 - a13*a21*a30) / d;
    Y[2][1] = (a00*a23*a31 - a00*a21*a33 + a01*a20*a33 - a01*a23*a30 - a03*a20*a31 + a03*a21*a30) / d;
    Y[2][2] = (a00*a11*a33 - a00*a13*a31 - a01*a10*a33 + a01*a13*a30 + a03*a10*a31 - a03*a11*a30) / d;
    Y[2][3] = (a00*a13*a21 - a00*a11*a23 + a01*a10*a23 - a01*a13*a20 - a03*a10*a21 + a03*a11*a20) / d;
    Y[3][0] = (a10*a22*a31 - a10*a21*a32 + a11*a20*a32 - a11*a22*a30 - a12*a20*a31 + a12*a21*a30) / d;
    Y[3][1] = (a00*a21*a32 - a00*a22*a31 - a01*a20*a32 + a01*a22*a30 + a02*a20*a31 - a02*a21*a30) / d;
    Y[3][2] = (a00*a12*a31 - a00*a11*a32 + a01*a10*a32 - a01*a12*a30 - a02*a10*a31 + a02*a11*a30) / d;
    Y[3][3] = (a00*a11*a22 - a00*a12*a21 - a01*a10*a22 + a01*a12*a20 + a02*a10*a21 - a02*a11*a20) / d;
  }

  else {
    fprintf(stderr, "Error: inv() not supported for > 4x4 matrices\n");
    exit(1);
  }
}

/**
 * Solves the quadratic equation: f(x) = a*x^2 + b*x + c
 * Sets x[0] and x[1] to be the real roots of f(x), if found.
 * Returns the number of real roots found.
 */
int solve_quadratic(double *x, double a, double b, double c)
{
  double s2 = b*b - 4*a*c;
  if (s2 < 0.0)
    return 0;

  double s = sqrt(s2);
  x[0] = .5*(-b + s)/a;
  x[1] = .5*(-b - s)/a;

  return 2;
}

/**
 * Solves the cubic equation: f(x) = a*x^3 + b*x^2 + c*x + d
 * Sets x[0], x[1], and x[2] to be the real roots of f(x), if found.
 * Returns the number of real roots found.
 *
int solve_cubic(double *x, double a, double b, double c, double d)
{
  double p = -b/(3.0*a);
  double q = p*p*p + (b*c - 3.0*a*d)/(6*a*a);
  double r = c/(3.0*a);
    
}
*/


/**
 * Compute the eigenvalues z and eigenvectors V of a real symmetric n-by-n matrix X
 * The eigenvalues, z, will be sorted from smallest to largest in magnitude, and the
 * eigenvectors will be stored in the rows of V.
 * @param X (input) Symmetric n-by-n matrix
 * @param n (input) Dimensionality of X
 * @param z (output) Eigenvalues of X, from smallest to largest in magnitude
 * @param V (output) Eigenvectors of X, in the rows
 */

/*
void eigen_symm(double z[], double **V, double **X, int n)
{
  double **Vt = matrix_clone(X,n,n);
  double z2[n];
  int i, j;

  //TODO: replace this with solve_cubic() for n < 4

  int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', n, Vt[0], n, z2);
  if (info)
    fprintf(stderr, "Error: eigen_symm failed to converge!\n");

  // sort eigenvalues
  double tolerance = 1e-10;
  int idx[n];
  sort_indices(z2, idx, n);
  if (z2[idx[0]] < -tolerance)  // negative eigenvalues --> sort in reverse order
    reversei(idx, idx, n);
  for (i = 0; i < n; i++)
    z[i] = z2[idx[i]];
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      V[i][j] = Vt[j][idx[i]];

  //cleanup
  free_matrix2(Vt);
}
*/


void eigen_symm_2d(double z[], double **V, double **X)
{
  double a = X[0][0];
  double b = X[0][1];
  double c = X[1][1];

  const double epsilon = 1e-16;

  if (b*b < epsilon * fabs(a*c)) {
    if (fabs(a) < fabs(c)) {
      z[0] = a;
      z[1] = c;
      V[0][0] = V[1][1] = 1.0;
      V[0][1] = V[1][0] = 0.0;
    }
    else {
      z[0] = c;
      z[1] = a;
      V[0][0] = V[1][1] = 0.0;
      V[0][1] = V[1][0] = 1.0;
    }
    return;
  }

  double s = sqrt((a+c)*(a+c) + 4.*(b*b-a*c));
  double z1 = (a+c+s)/2.;
  double z2 = (a+c-s)/2.;
  if (fabs(z1) < fabs(z2)) {
    z[0] = z1;
    z[1] = z2;
  }
  else {
    z[1] = z1;
    z[0] = z2;
  }

  double d0 = hypot(b, z[0]-a);
  double d1 = hypot(b, z[1]-a);
  V[0][0] = b/d0;
  V[0][1] = (z[0]-a)/d0;
  V[1][0] = b/d1;
  V[1][1] = (z[1]-a)/d1;
}


void eigen_symm(double z[], double **V, double **X, int n)
{
  if (n == 2) {
    eigen_symm_2d(z,V,X);
    return;
  }

  // naive Jacobi method
  int i, j;
  double tolerance = 1e-10;
  double **A = matrix_clone(X,n,n);
  double **B = new_matrix2(n,n);
  double **G = new_matrix2(n,n);
  double **Gt = new_matrix2(n,n);

  int cnt = 1; //dbug

  // initialize V = I
  for (i = 0; i < n; i++) {
    V[i][i] = 1;
    for (j = i+1; j < n; j++)
      V[i][j] = V[j][i] = 0;
  }

  //printf("break 1\n");  //dbug

  while (1) {

    //dbug
    //printf("A:\n");
    //print_matrix(A, n, n);
    //printf("\n");

    // check for convergence
    double d_off = 0, d_diag = 0;
    for (i = 0; i < n; i++) {
      d_diag += A[i][i]*A[i][i];
      for (j = i+1; j < n; j++)
	d_off = MAX(d_off, fabs(A[i][j]));
    }
    d_diag = sqrt(d_diag / (double)n);
    if (d_off < MAX(tolerance * d_diag, tolerance))
      break;

    //dbug
    if (cnt++ % 1000 == 0) {
      printf("d_off = %e, d_diag = %e\n", d_off, d_diag);  //dbug
      if (!isfinite(d_off) || !isfinite(d_diag))
	return;
    }

    // find largest pivot
    double pivot = 0;
    int ip=0, jp=0;
    for (i = 0; i < n; i++) {
      for (j = i+1; j < n; j++) {
	double p = fabs(A[i][j]);
	if (p > pivot) {
	  pivot = p;
	  ip = i;
	  jp = j;
	}
      }
    }

    //printf("pivot = %f, ip = %d, jp = %d\n", pivot, ip, jp);  //dbug
    
    // compute Givens cos, sin
    double a = (A[jp][jp] - A[ip][ip]) / (2 * A[ip][jp]);
    double t = 1 / (fabs(a) + sqrt(1 + a*a));  // tan
    if (a < 0)
      t = -t;
    double c = 1 / sqrt(1 + t*t);  // cos
    double s = t*c;  // sin

    //printf("a = %f, t = %f, c = %f, s = %f\n", a, t, c, s);  //dbug

    // compute Givens rotation matrix
    for (i = 0; i < n; i++) {
      G[i][i] = 1;
      for (j = i+1; j < n; j++)
	G[i][j] = G[j][i] = 0;
    }
    G[ip][ip] = G[jp][jp] = c;
    G[ip][jp] = s;
    G[jp][ip] = -s;

    //dbug
    //printf("givens rotation matrix:\n");
    //print_matrix(G, n, n);
    //printf("\n");
    
    // compute new A
    transpose(Gt, G, n, n);
    matrix_mult(B, A, G, n, n, n);   // B = A*G
    matrix_mult(A, Gt, B, n, n, n);  // A = Gt*B
    
    // compute new V (with eigenvectors in the rows)
    matrix_mult(B, Gt, V, n, n, n);  // B = Gt*V
    matrix_copy(V, B, n, n);     // V = B;
  }

  //printf("break 2\n");  //dbug

  // sort eigenvalues
  int idx[n];
  double z2[n];
  for (i = 0; i < n; i++)
    z[i] = A[i][i];
  sort_indices(z, idx, n);
  if (z[idx[0]] < -tolerance) {  // negative eigenvalues --> sort in reverse order
    reversei(idx, idx, n);
    //for (i = 0; i < n/2; i++) {
    //  int tmp = idx[i];
    //  idx[i] = idx[n-i-1];
    //  idx[n-i-1] = tmp;
    //}
  }
  for (i = 0; i < n; i++)
    z2[i] = z[idx[i]];
  for (i = 0; i < n; i++)
    z[i] = z2[i];
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      B[i][j] = V[idx[i]][j];
  matrix_copy(V, B, n, n);

  free_matrix2(A);
  free_matrix2(B);
  free_matrix2(G);
  free_matrix2(Gt);
}


// reorder the rows of X, Y = X(idx,:)
void reorder_rows(double **Y, double **X, int *idx, int n, int m)
{
  int i;
  double **Y2 = (X==Y ? new_matrix2(n,m) : Y);
  for (i = 0; i < n; i++)
    memcpy(Y2[i], X[idx[i]], m*sizeof(double));
  if (X==Y) {
    matrix_copy(Y, Y2, n, m);
    free_matrix2(Y2);
  }
}

// reorder the rows of X, Y = X(idx,:)
void reorder_rowsi(int **Y, int **X, int *idx, int n, int m)
{
  int i;
  int **Y2 = (X==Y ? new_matrix2i(n,m) : Y);
  for (i = 0; i < n; i++)
    memcpy(Y2[i], X[idx[i]], m*sizeof(int));
  if (X==Y) {
    memcpy(Y[0], Y2[0], n*m*sizeof(int));
    free_matrix2i(Y2);
  }
}

void repmat(double **B, double **A, int rep_n, int rep_m, int n, int m)
{
  int i, rep_i, rep_j;

  // copy A into top-left corner of B
  if (A != B)
    for (i = 0; i < n; ++i)
      memcpy(B[i], A[i], m * sizeof(double)); 

  // repeat top-left corner to the right
  for (i = 0; i < n; ++i)
    for (rep_j = 1; rep_j < rep_m; ++rep_j)
      memcpy(&B[i][rep_j * m], B[i], m * sizeof(double)); 

  // repeat top of B downwards
  for (rep_i = 1; rep_i < rep_n; ++rep_i)
    memcpy(B[rep_i * n], B[0], m * rep_m * n * sizeof(double));
}

void repmati(int **B, int **A, int rep_n, int rep_m, int n, int m)
{
  int i, rep_i, rep_j;

  // copy A into top-left corner of B
  if (A != B)
    for (i = 0; i < n; ++i)
      memcpy(B[i], A[i], m * sizeof(int)); 

  // repeat top-left corner to the right
  for (i = 0; i < n; ++i)
    for (rep_j = 1; rep_j < rep_m; ++rep_j)
      memcpy(&B[i][rep_j * m], B[i], m * sizeof(int)); 

  // repeat top of B downwards
  for (rep_i = 1; rep_i < rep_n; ++rep_i)
    memcpy(B[rep_i * n], B[0], m * rep_m * n * sizeof(int));
}

/*
 * blur matrix with a 3x3 gaussian filter with sigma=.5
 */
void blur_matrix(double **dst, double **src, int n, int m)
{
  double G[3] = {.6193, .0838, .0113};

  double **I = (dst==src ? new_matrix2(n,m) : dst);
  memcpy(I[0], src[0], n*m*sizeof(double));

  int i,j;
  for (i = 1; i < n-1; i++)
    for (j = 1; j < m-1; j++)
      I[i][j] = G[0]*src[i][j] + G[1]*(src[i+1][j] + src[i-1][j] + src[i][j+1] + src[i][j-1]) + G[2]*(src[i+1][j+1] + src[i+1][j-1] + src[i-1][j+1] + src[i-1][j-1]);

  if (dst==src) {
    memcpy(dst[0], I[0], n*m*sizeof(double));
    free_matrix2(I);
  }
}

/*
 * blur masked matrix with a 3x3 gaussian filter with sigma=.5
 */
void blur_matrix_masked(double **dst, double **src, int **mask, int n, int m)
{
  double G[3] = {.6193, .0838, .0113};

  double **I = (dst==src ? new_matrix2(n,m) : dst);
  memcpy(I[0], src[0], n*m*sizeof(double));

  int i,j;
  for (i = 1; i < n-1; i++) {
    for (j = 1; j < m-1; j++) {
      double x[9] = {mask[i][j] ? src[i][j] : 0., mask[i+1][j] ? src[i+1][j] : 0., mask[i-1][j] ? src[i-1][j] : 0., mask[i][j+1] ? src[i][j+1] : 0., mask[i][j-1] ? src[i][j-1] : 0.,
		  mask[i+1][j+1] ? src[i+1][j+1] : 0., mask[i+1][j-1] ? src[i+1][j-1] : 0., mask[i-1][j+1] ? src[i-1][j+1] : 0., mask[i-1][j-1] ? src[i-1][j-1] : 0.};

      double v = G[0]*x[0] + G[1]*(x[1] + x[2] + x[3] + x[4]) + G[2]*(x[5] + x[6] + x[7] + x[8]);

      int n0 = !!mask[i][j];
      int n1 = !!mask[i+1][j] + !!mask[i-1][j] + !!mask[i][j+1] + !!mask[i][j-1];
      int n2 = !!mask[i+1][j+1] + !!mask[i-1][j-1] + !!mask[i-1][j+1] + !!mask[i+1][j-1];
      double g_tot = n0*G[0] + n1*G[1] + n2*G[2];

      I[i][j] = v / g_tot;
    }
  }

  if (dst==src) {
    memcpy(dst[0], I[0], n*m*sizeof(double));
    free_matrix2(I);
  }
}

void print_matrix(double **X, int n, int m)
{
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++)
      printf("%f ", X[i][j]);
    printf("\n");
  }
}


// perform linear regression: dot(b,x[i]) = y[i], i=1..n
void linear_regression(double *b, double **X, double *y, int n, int d)
{
  double **Xt = new_matrix2(d,n);
  transpose(Xt,X,n,d);

  double **XtX = new_matrix2(d,d);
  matrix_mult(XtX,Xt,X,d,n,d);

  double Xty[d];
  matrix_vec_mult(Xty,Xt,y,d,n);

  solve(b, XtX, Xty, d);

  free_matrix2(Xt);
  free_matrix2(XtX);
}


// fit a polynomial: \sum{b[i]*x[j]^i} = y[j], i=1..n, j=1..d
void polynomial_regression(double *b, double *x, double *y, int n, int d)
{
  double **X = new_matrix2(n,d);

  int i, j;
  for (i = 0; i < n; i++)
    X[i][0] = 1;

  for (i = 0; i < n; i++)
    for (j = 1; j < d; j++)
      X[i][j] = X[i][j-1]*x[i];

  linear_regression(b,X,y,n,d);

  free_matrix2(X);
}






/* create a new graph
graph_t *graph_new(int num_vertices, int edge_capacity)
{
  int i;
  graph_t *g = (graph_t *)malloc(sizeof(graph_t));

  g->nv = num_vertices;
  g->vertices = (vertex_t *)calloc(g->nv, sizeof(vertex_t));
  for (i = 0; i < g->nv; i++)
    g->vertices[i].index = i;

  g->ne = 0;
  g->_edge_capacity = edge_capacity;
  g->edges = (edge_t *)calloc(edge_capacity, sizeof(edge_t));

  return g;
}
*/


// free a graph
void graph_free(graph_t *g)
{
  int i;
  free(g->edges);
  for (i = 0; i < g->nv; i++) {
    free(g->vertices[i].edges);
    ilist_free(g->vertices[i].neighbors);
  }
  free(g);
}


/* add an edge to a graph
void graph_add_edge(graph_t *g, int i, int j)
{
  ilist
  for 


  if (g->ne == g->_edge_capacity) {
    g->_edge_capacity *= 2;
    g->edges = (edge_t *)realloc(g->edges, g->_edge_capacity * sizeof(edge_t));
  }

  g->edges[g->ne].i = i;
  g->edges[g->ne].j = j;
  g->ne++;

  g->vertices[i]
}
*/


// find the index of an edge in a graph
int graph_find_edge(graph_t *g, int i, int j)
{
  int k = ilist_find(g->vertices[i].neighbors, j);

  if (k < 0)
    return -1;

  return g->vertices[i].edges[k];
}


// smooth the edges of a graph
void graph_smooth(double **dst, double **src, graph_t *g, int d, double w)
{
  int i, j;
  double p[d];
  ilist_t *v;

  if (dst != src)
    memcpy(dst[0], src[0], d*g->nv*sizeof(double));

  for (i = 0; i < g->nv; i++) {
    memset(p, 0, d * sizeof(double));                        // p = 0
    for (v = g->vertices[i].neighbors; v; v = v->next) {
      j = v->x;
      add(p, p, dst[j], d);                                  // p += dst[j]
    }
    mult(p, p, 1/norm(p, d), d);                             // p = p/norm(p)

    wavg(dst[i], p, dst[i], w, d);                           // dst[i] = w*p + (1-w)*dst[i]
  }
}


static int dcomp(const void *px, const void *py)
{
  double x = *(double *)px;
  double y = *(double *)py;

  if (x == y)
    return 0;

  return (x < y ? -1 : 1);
}

// sample uniformly from a simplex S with n vertices
void sample_simplex(double x[], double **S, int n, int d)
{
  int i;

  // get n-1 uniform samples, u, on [0,1], and sort them
  double u[n];
  for (i = 0; i < n-1; i++)
    u[i] = frand();
  u[n-1] = 1;

  qsort((void *)u, n-1, sizeof(double), dcomp);

  // mixing coefficients are the order statistics of u
  double c[n];
  c[0] = u[0];
  for (i = 1; i < n; i++)
    c[i] = u[i] - u[i-1];

  // x = sum(c[i]*S[i])
  mult(x, S[0], c[0], d);
  for (i = 1; i < n; i++) {
    double y[d];
    mult(y, S[i], c[i], d);
    add(x, x, y, d);
  }
}


/*******************
typedef struct {
  int nv;
  int ne;
  int nf;
  int *vertices;
  edge_t *edges;
  face_t *faces;
  int *nvn;  // # vertex neighbors
  int *nen;  // # edge neighbors
  int **vertex_neighbors;  // vertex -> {vertices}
  int **vertex_edges;      // vertex -> {edges}
  int **edge_neighbors;    // edge -> {vertices}
  int **edge_faces;        // edge -> {faces}
  // internal vars
  int _vcap;
  int _ecap;
  int _fcap;
  int *_vncap;
  int *_encap;
} meshgraph_t;
******************/


/*
 * Create a new meshgraph with initial vertex capacity 'vcap' and degree capacity 'dcap'.
 */
meshgraph_t *meshgraph_new(int vcap, int dcap)
{
  int i;
  meshgraph_t *g;
  safe_calloc(g, 1, meshgraph_t);

  safe_malloc(g->vertices, vcap, int);
  safe_malloc(g->edges, vcap, edge_t);
  safe_malloc(g->faces, vcap, face_t);

  g->_vcap = g->_ecap = g->_fcap = vcap;
  safe_malloc(g->_vncap, vcap, int);
  safe_malloc(g->_encap, vcap, int);

  safe_malloc(g->vertex_neighbors, vcap, int *);
  safe_malloc(g->vertex_edges, vcap, int *);
  safe_calloc(g->nvn, vcap, int);
  for (i = 0; i < vcap; i++) {
    safe_malloc(g->vertex_neighbors[i], dcap, int);
    safe_malloc(g->vertex_edges[i], dcap, int);
    g->_vncap[i] = dcap;
  }

  safe_malloc(g->edge_neighbors, vcap, int *);
  safe_malloc(g->edge_faces, vcap, int *);
  safe_calloc(g->nen, vcap, int);
  for (i = 0; i < vcap; i++) {
    safe_malloc(g->edge_neighbors[i], dcap, int);
    safe_malloc(g->edge_faces[i], dcap, int);
    g->_encap[i] = dcap;
  }

  return g;
}


void meshgraph_free(meshgraph_t *g)
{
  int i;

  free(g->vertices);
  free(g->edges);
  free(g->faces);
  free(g->nvn);
  free(g->nen);
  free(g->_vncap);
  free(g->_encap);

  for (i = 0; i < g->nv; i++) {
    free(g->vertex_neighbors[i]);
    free(g->vertex_edges[i]);
  }
  free(g->vertex_neighbors);
  free(g->vertex_edges);

  for (i = 0; i < g->ne; i++) {
    free(g->edge_neighbors[i]);
    free(g->edge_faces[i]);
  }
  free(g->edge_neighbors);
  free(g->edge_faces);

  free(g);
}


int meshgraph_find_edge(meshgraph_t *g, int i, int j)
{
  int n;
  for (n = 0; n < g->nvn[i]; n++)
    if (g->vertex_neighbors[i][n] == j)
      return g->vertex_edges[i][n];

  return -1;
}


int meshgraph_find_face(meshgraph_t *g, int i, int j, int k)
{
  int e = meshgraph_find_edge(g, i, j);
  if (e < 0)
    return -1;

  int n;
  for (n = 0; n < g->nen[e]; n++)
    if (g->edge_neighbors[e][n] == k)
      return g->edge_faces[e][n];

  return -1;
}


static inline int meshgraph_add_vertex_neighbor(meshgraph_t *g, int i, int vertex, int edge)
{
  int n = g->nvn[i];
  if (n == g->_vncap[i]) {
    g->_vncap[i] *= 2;
    safe_realloc(g->vertex_neighbors[i], g->_vncap[i], int);
    safe_realloc(g->vertex_edges[i], g->_vncap[i], int);
  }
  g->vertex_neighbors[i][n] = vertex;
  g->vertex_edges[i][n] = edge;
  g->nvn[i]++;

  return n;
}


int meshgraph_add_edge(meshgraph_t *g, int i, int j)
{
  //printf("meshgraph_add_edge(%d, %d)\n", i, j);

  int edge = meshgraph_find_edge(g, i, j);
  if (edge >= 0)
    return edge;

  //printf("  break 1\n");

  // add the edge
  if (g->ne == g->_ecap) {
    int old_ecap = g->_ecap;
    g->_ecap *= 2;
    safe_realloc(g->edges, g->_ecap, edge_t);
    safe_realloc(g->edge_neighbors, g->_ecap, int *);
    safe_realloc(g->edge_faces, g->_ecap, int *);
    safe_realloc(g->nen, g->_ecap, int);
    safe_realloc(g->_encap, g->_ecap, int);

    //printf("  break 1.1\n");

    int e;
    for (e = old_ecap; e < g->_ecap; e++) {
      //printf("    e = %d\n", e);
      g->nen[e] = 0;
      int dcap = g->_encap[0];

      //printf("    dcap = %d\n", dcap);

      safe_malloc(g->edge_neighbors[e], dcap, int);

      //printf("    break 1.1.1\n");

      safe_malloc(g->edge_faces[e], dcap, int);

      //printf("    break 1.1.2\n");

      g->_encap[e] = dcap;
    }
  }

  //printf("  break 2\n");

  edge = g->ne;
  g->edges[edge].i = i;
  g->edges[edge].j = j;
  g->ne++;

  //printf("  break 3\n");

  // add the vertex neighbors
  meshgraph_add_vertex_neighbor(g, i, j, edge);
  meshgraph_add_vertex_neighbor(g, j, i, edge);

  //printf("  break 4\n");

  return edge;
}


static inline int meshgraph_add_edge_neighbor(meshgraph_t *g, int i, int vertex, int face)
{
  int n = g->nen[i];

  //printf("g->nen[%d] = %d, g->_encap[%d] = %d\n", i, n, i, g->_encap[i]);

  if (n == g->_encap[i]) {

    //printf("n == g->_encap[%d]\n", i);

    g->_encap[i] *= 2;
    safe_realloc(g->edge_neighbors[i], g->_encap[i], int);
    safe_realloc(g->edge_faces[i], g->_encap[i], int);
  }

  //printf("  break 1\n");

  g->edge_neighbors[i][n] = vertex;

  //printf("  break 2\n");

  g->edge_faces[i][n] = face;

  //printf("  break 3\n");

  g->nen[i]++;

  return n;
}


int meshgraph_add_face(meshgraph_t *g, int i, int j, int k)
{
  //printf("meshgraph_add_face(%d, %d, %d)\n", i, j, k);

  int face = meshgraph_find_face(g, i, j, k);
  if (face >= 0)
    return face;

  //printf("  break 1\n");

  // add the edges
  int edge_ij = meshgraph_add_edge(g, i, j);
  int edge_ik = meshgraph_add_edge(g, i, k);
  int edge_jk = meshgraph_add_edge(g, j, k);

  //printf("  break 2\n");

  // add the face
  //printf("g->nf = %d, g->_fcap = %d\n", g->nf, g->_fcap);

  if (g->nf == g->_fcap) {
    g->_fcap *= 2;
    safe_realloc(g->faces, g->_fcap, face_t);
  }
  face = g->nf;
  g->faces[face].i = i;
  g->faces[face].j = j;
  g->faces[face].k = k;
  g->nf++;

  //printf("  break 3\n");

  // add the edge neighbors
  meshgraph_add_edge_neighbor(g, edge_ij, k, face);
  meshgraph_add_edge_neighbor(g, edge_ik, j, face);
  meshgraph_add_edge_neighbor(g, edge_jk, i, face);

  //printf("  break 4\n");

  return face;
}


static int _sortable_cmp(const void *x1, const void *x2)
{
  double v1 = ((sortable_t *)x1)->value;
  double v2 = ((sortable_t *)x2)->value;

  if (v1 == v2)
    return 0;

  if (v1 < v2 || isnan(v1))
    return -1;

  return 1;
}

// sort an array of weighted data using qsort
void sort_data(sortable_t *x, size_t n)
{
  qsort(x, n, sizeof(sortable_t), _sortable_cmp);
}


// sort the indices of x (leaving x unchanged)
void sort_indices(double *x, int *idx, int n)
{
  int i;
  sortable_t *s;
  int *xi;
  safe_malloc(s, n, sortable_t);
  safe_malloc(xi, n, int);

  for (i = 0; i < n; i++) {
    xi[i] = i;
    s[i].value = x[i];
    s[i].data = (void *)(&xi[i]);
  }

  sort_data(s, n);

  for (i = 0; i < n; i++)
    idx[i] = *(int *)(s[i].data);

  free(s);
  free(xi);
}


/*
 * fills idx with the indices of the k min entries of x
 *   --> works by maintaining the invariant that idx[0] always has the
 *       largest x value of the min k found so far
 */       
void mink(double *x, int *idx, int n, int k)
{
  int i, j;

  // initialize idx with 1:k
  for (i = 0; i < k; i++)
    idx[i] = i;

  // maintain invariant
  for (i = 1; i < k; i++) {
    if (x[idx[i]] > x[idx[0]]) {
      int tmp = idx[i];
      idx[i] = idx[0];
      idx[0] = tmp;
    }
  }

  // process the rest of x
  for (i = k; i < n; i++) {
    if (x[i] < x[idx[0]]) {
      idx[0] = i;

      // maintain invariant
      for (j = 1; j < k; j++) {
	if (x[idx[j]] > x[idx[0]]) {
	  int tmp = idx[j];
	  idx[j] = idx[0];
	  idx[0] = tmp;
	}
      }
    }
  }

  // sort the min k values
  double xmink[k];
  int idx2[k];
  for (i = 0; i < k; i++)
    xmink[i] = x[idx[i]];
  sort_indices(xmink, idx2, k);
  for (i = 0; i < k; i++)
    idx2[i] = idx[idx2[i]];
  for (i = 0; i < k; i++)
    idx[i] = idx2[i];
}

short double_is_equal(double a, double b)
{
  return fabs(a - b) < 0.00001;
}

// fast select algorithm
int qselect(double *x, int n, int k)
{
  if (n == 1)
    return 0;

  double pivot = x[k];

  // partition x into y < pivot, z > pivot
  int i, ny=0, nz=0;
  for (i = 0; i < n; i++) {
    if (x[i] < pivot)
      ny++;
    else if (x[i] > pivot)
      nz++;
  }

  if (k < ny) {
    double *y;
    int *yi;
    safe_calloc(y, ny, double);
    safe_calloc(yi, ny, int);
    ny = 0;
    for (i = 0; i < n; i++) {
      if (x[i] < pivot) {
	yi[ny] = i;
	y[ny++] = x[i];
      }
    }
    i = yi[qselect(y, ny, k)];
    free(y);
    free(yi);
    return i;
  }

  else if (k >= n - nz) {
    double *z;
    int *zi;
    safe_calloc(z, nz, double);
    safe_calloc(zi, nz, int);
    nz = 0;
    for (i = 0; i < n; i++) {
      if (x[i] > pivot) {
	zi[nz] = i;
	z[nz++] = x[i];
      }
    }
    i = zi[qselect(z, nz, k-(n-nz))];
    free(z);
    free(zi);
    return i;
  }

  return k;
}


static kdtree_t *build_kdtree(double **X, int *xi, int n, int d, int depth)
{
  if (n == 0)
    return NULL;

  int i, axis = depth % d;
  kdtree_t *node;
  safe_calloc(node, 1, kdtree_t);
  node->axis = axis;

  double *x;
  safe_calloc(x, n, double);
  for (i = 0; i < n; i++)
    x[i] = X[i][axis];

  int median = qselect(x, n, n/2);

  // node location
  node->i = xi[median];
  node->d = d;
  safe_malloc(node->x, d, double);
  memcpy(node->x, X[median], d*sizeof(double));

  // node bbox init:  bbox_min = bbox_max = x
  safe_malloc(node->bbox_min, d, double);
  safe_malloc(node->bbox_max, d, double);
  memcpy(node->bbox_min, node->x, d*sizeof(double));
  memcpy(node->bbox_max, node->x, d*sizeof(double));

  // partition x into y < pivot, z > pivot
  double pivot = x[median];
  int ny=0, nz=0;
  for (i = 0; i < n; i++) {
    if (i == median)
      continue;
    if (x[i] <= pivot)
      ny++;
    else if (x[i] > pivot)
      nz++;
  }

  //printf("n = %d, d = %d, depth = %d, axis = %d --> median = %d, X[median] = (%f, %f, %f), ny = %d, nz = %d\n",
  //	 n, d, depth, axis, median, X[median][0], X[median][1], X[median][2], ny, nz);


  if (ny > 0) {
    double **Y = new_matrix2(ny, d);
    int *yi;
    safe_calloc(yi, ny, int);
    ny = 0;
    for (i = 0; i < n; i++) {
      if (i == median)
	continue;
      if (x[i] <= pivot) {
	yi[ny] = xi[i];
	memcpy(Y[ny], X[i], d*sizeof(double));
	ny++;
      }
    }
    node->left = build_kdtree(Y, yi, ny, d, depth+1);

    // update bbox
    for (i = 0; i < d; i++) {
      if (node->left->bbox_min[i] < node->bbox_min[i])
	node->bbox_min[i] = node->left->bbox_min[i];
      if (node->left->bbox_max[i] > node->bbox_max[i])
	node->bbox_max[i] = node->left->bbox_max[i];
    }

    free_matrix2(Y);
    free(yi);
  }

  if (nz > 0) {
    double **Z = new_matrix2(nz, d);
    int *zi;
    safe_calloc(zi, nz, int);
    nz = 0;
    for (i = 0; i < n; i++) {
      if (i == median)
	continue;
      if (x[i] > pivot) {
	zi[nz] = xi[i];
	memcpy(Z[nz], X[i], d*sizeof(double));
	nz++;
      }
    }
    node->right = build_kdtree(Z, zi, nz, d, depth+1);

    // update bbox
    for (i = 0; i < d; i++) {
      if (node->right->bbox_min[i] < node->bbox_min[i])
	node->bbox_min[i] = node->right->bbox_min[i];
      if (node->right->bbox_max[i] > node->bbox_max[i])
	node->bbox_max[i] = node->right->bbox_max[i];
    }

    free_matrix2(Z);
    free(zi);
  }

  free(x);
  return node;
}


kdtree_t *kdtree(double **X, int n, int d)
{
  int i, *xi;
  safe_malloc(xi, n, int);
  for (i = 0; i < n; i++)
    xi[i] = i;

  kdtree_t *tree = build_kdtree(X, xi, n, d, 0);

  free(xi);
  return tree;
}


static kdtree_t *kdtree_NN_node(kdtree_t *tree, double *x, kdtree_t *best)
{
  if (tree == NULL)
    return best;

  //printf("node %d", tree->i);

  int i, d = tree->d;
  double dbest = (best ? dist(x, best->x, d) : DBL_MAX);

  // first, check if any node in tree can possibly be better than 'best'
  if (best) {
    double y[d];  // closest point on the tree's bbox to x
    for (i = 0; i < d; i++) {
      if (x[i] < tree->bbox_min[i])
	y[i] = tree->bbox_min[i];
      else if (x[i] > tree->bbox_max[i])
	y[i] = tree->bbox_max[i];
      else
	y[i] = x[i];
    }
    if (dist(y, x, d) >= dbest) {  // 'best' is closer than the closest possible point in tree, so return
      //printf("  --> pruned!\n");
      return best;
    }
  }

  int axis = tree->axis;
  kdtree_t *nn = best;

  // compare with the node itself
  double dtree = dist(x, tree->x, d);
  //printf(" (%f)", dtree);
  if (dtree < dbest) {
    nn = tree;
    dbest = dtree;
    //printf(" --> new best");
  }
  //printf("\n");

  // compare with the NN in each sub-tree
  if (x[axis] <= tree->x[axis]) {
    nn = kdtree_NN_node(tree->left, x, nn);
    nn = kdtree_NN_node(tree->right, x, nn);
  }
  else if (x[axis] > tree->x[axis]) {
    nn = kdtree_NN_node(tree->right, x, nn);
    nn = kdtree_NN_node(tree->left, x, nn);
  }

  //dbest = dist(x, nn->x, d);
  //printf(" ... return %d (%f)\n", nn->i, dbest);

  return nn;
}

int kdtree_NN(kdtree_t *tree, double *x)
{
  kdtree_t *nn = kdtree_NN_node(tree, x, NULL);
  return (nn ? nn->i : -1);
}


void kdtree_free(kdtree_t *tree)
{
  if (tree == NULL)
    return;

  kdtree_free(tree->left);
  kdtree_free(tree->right);

  free(tree->x);
  free(tree->bbox_min);
  free(tree->bbox_max);

  free(tree);
}

// RGB to CIELAB color space
void rgb2lab(double lab[], double rgb[])
{
  double R = rgb[0];
  double G = rgb[1];
  double B = rgb[2];

  //if (R > 1.0 || G > 1.0 || B > 1.0) {
  R /= 255.0;
  G /= 255.0;
  B /= 255.0;
  //}
  
  // set a threshold
  double T = 0.008856;
  
  // RGB to XYZ
  double X = 0.412453*R + 0.357580*G + 0.180423*B;
  double Y = 0.212671*R + 0.715160*G + 0.072169*B;
  double Z = 0.019334*R + 0.119193*G + 0.950227*B;

  // normalize for D65 white point
  X /= 0.950456;
  Z /= 1.088754;

  double X3 = pow(X, 1/3.);
  double Y3 = pow(Y, 1/3.);
  double Z3 = pow(Z, 1/3.);

  double fX = (X>T ? X3 : 7.787*X + 16/116.);
  double fY = (Y>T ? Y3 : 7.787*Y + 16/116.);
  double fZ = (Z>T ? Z3 : 7.787*Z + 16/116.);

  lab[0] = (Y>T ? 116*Y3 - 16.0 : 903.3*Y);
  lab[1] = 500*(fX - fY);
  lab[2] = 200*(fY - fZ);
}

// CIELAB to RGB color space
void lab2rgb(double rgb[], double lab[])
{
  // Thresholds
  double T1 = 0.008856;
  double T2 = 0.206893;

  double L = lab[0];
  double a = lab[1];
  double b = lab[2];

  // Compute Y
  double fY = pow((L + 16) / 116., 3);
  int YT = (fY > T1);
  if (!YT)
    fY = L / 903.3;
  double Y = fY;

  // Alter fY slightly for further calculations
  fY = (YT ? pow(fY, 1/3.) : 7.787 * fY + 16/116.);

  // Compute X
  double fX = a / 500. + fY;
  int XT = fX > T2;
  double X = (XT ? pow(fX, 3) : (fX - 16/116.) / 7.787);

  // Compute Z
  double fZ = fY - b / 200.;
  int ZT = fZ > T2;
  double Z = (ZT ? pow(fZ, 3) : (fZ - 16/116.) / 7.787);

  // Normalize for D65 white point
  X = X * 0.950456;
  Z = Z * 1.088754;

  // XYZ to RGB
  double R =  3.240479*X - 1.537150*Y - 0.498535*Z;
  double G = -0.969256*X + 1.875992*Y + 0.041556*Z;
  double B =  0.055648*X - 0.204043*Y + 1.057311*Z;

  rgb[0] = 255*MAX(MIN(R, 1.0), 0.0);
  rgb[1] = 255*MAX(MIN(G, 1.0), 0.0);
  rgb[2] = 255*MAX(MIN(B, 1.0), 0.0);
}
