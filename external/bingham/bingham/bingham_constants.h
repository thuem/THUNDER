
#ifndef BINGHAM_CONSTANTS_H
#define BINGHAM_CONSTANTS_H

extern const double BINGHAM_MIN_CONCENTRATION;


void bingham_constants_init();
void bingham_dY_params_3d(double *Z, double *F, double *dY);
double bingham_F_lookup_3d(double *Z);
void bingham_dF_lookup_3d(double *dF, double *Z);

double bingham_F_table_get(int i, int j, int k);
double bingham_dF1_table_get(int i, int j, int k);
double bingham_dF2_table_get(int i, int j, int k);
double bingham_dF3_table_get(int i, int j, int k);
double bingham_dF_table_get(int a, int i, int j, int k);


//---------------- Bingham normalizing constants F(z) and partial derivatives ------------------//

double bingham_F_1d(double z);
double bingham_dF_1d(double z);
double bingham_F_2d(double z1, double z2);
double bingham_dF1_2d(double z1, double z2);
double bingham_dF2_2d(double z1, double z2);
double bingham_F_3d(double z1, double z2, double z3);
double bingham_dF1_3d(double z1, double z2, double z3);
double bingham_dF2_3d(double z1, double z2, double z3);
double bingham_dF3_3d(double z1, double z2, double z3);


//----------------- Bingham F(z) "compute_all" tools --------------------//

void compute_all_bingham_F_2d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step);
void compute_all_bingham_dF1_2d(double z1_min, double z1_max, double z1_step,
				double z2_min, double z2_max, double z2_step);
void compute_all_bingham_dF2_2d(double z1_min, double z1_max, double z1_step,
				double z2_min, double z2_max, double z2_step);
void compute_all_bingham_F_3d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step,
			      double z3_min, double z3_max, double z3_step);
void compute_all_bingham_dF1_3d(double z1_min, double z1_max, double z1_step,
				double z2_min, double z2_max, double z2_step,
				double z3_min, double z3_max, double z3_step);
void compute_all_bingham_dF2_3d(double z1_min, double z1_max, double z1_step,
				double z2_min, double z2_max, double z2_step,
				double z3_min, double z3_max, double z3_step);
void compute_all_bingham_dF3_3d(double z1_min, double z1_max, double z1_step,
				double z2_min, double z2_max, double z2_step,
				double z3_min, double z3_max, double z3_step);

void compute_range_bingham_F_2d(double *y, int n);
void compute_range_bingham_dF1_2d(double *y, int n);
void compute_range_bingham_dF2_2d(double *y, int n);
void compute_range_bingham_F_3d(double *y, int n, int k0, int k1);
void compute_range_bingham_dF1_3d(double *y, int n, int k0, int k1);
void compute_range_bingham_dF2_3d(double *y, int n, int k0, int k1);
void compute_range_bingham_dF3_3d(double *y, int n, int k0, int k1);



#endif
