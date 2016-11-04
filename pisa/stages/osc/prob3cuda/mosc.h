/*
 * author: Timothy C. Arlen
 *
 * date: 31 Jan 2014
 *
 * Broke off code from probGpu.cu to put in its own file, to make
 * analagous to original prob3++.
 *
 */

#ifndef __MOSCHINCLUDED__
#define __MOSCHINCLUDED__

#define elec (0)
#define muon (1)
#define tau  (2)
#define re (0)
#define im (1)

typedef enum nu_type {
  data_type,
  nue_type,
  numu_type,
  nutau_type,
  sterile_type,
  unknown_type} NuType;


typedef enum matrix_type {
  standard_type,
  barger_type} MatrixType;


__device__ void getM(fType Enu, fType rho,
                     fType Mix[][3][2], fType dmVacVac[][3], int antitype,
                     fType dmMatMat[][3], fType dmMatVac[][3]);
__device__ void getA(fType L, fType E, fType rho,
                     fType Mix[][3][2], fType dmMatVac[][3],
                     fType dmMatMat[][3], int antitype, fType A[3][3][2],
                     fType phase_offset);
__device__ void get_product(fType L, fType E, fType rho,fType Mix[][3][2],
                            fType dmMatVac[][3], fType dmMatMat[][3],
                            int antitype,
                            fType product[][3][3][2]);


#endif /* __MOSCHINCLUDED__ */
