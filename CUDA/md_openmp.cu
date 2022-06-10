#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../timer.h"

int main ( int argc, char *argv[] );
void compute ( int np, int nd, double pos[], double vel[], 
  double mass, double f[], double *pot, double *kin ,int j);
double dist ( int nd, double r1[], double r2[], double dr[] );
void initialize ( int np, int nd, double box[], int *seed, double pos[], 
  double vel[], double acc[] );
double r8_uniform_01 ( int *seed );
void timestamp ( );

#define gridSize 4
#define blockSize 1024
__global__ void update ( int np, int nd, double* pos, double* vel, double* f, double* acc, double mass, double dt )
{
 
  double rmass;

  rmass = 1.0 / mass;



  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  // printf("%8d\n",idx);
  while(idx < (np*nd) ){

    pos[idx] = pos[idx] + vel[idx] * dt + 0.5 * acc[idx] * dt * dt;
    vel[idx] = vel[idx] + 0.5 * dt * ( f[idx] * rmass + acc[idx] );
    acc[idx] = f[idx] * rmass;
    idx+=stride;
  }

  return;
}
/*********************************/
__global__ void compute_rd ( int np, int nd, double* pos,int j,double *d,double *rij){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  int k;
  if(idx>=np*nd)return;

  while(idx<np*nd){
    k = idx / nd;
    if(k!=j){
      rij[idx] = pos[idx] - pos[idx-nd*(k-j)];
      d[idx] = rij[idx]*rij[idx];
      


    }else{
      d[idx] = 0;
    }
    idx+=stride;
  }
}
__global__ void compute_d2 ( int np, int nd,double *d,double *d2,double *pe){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double PI2 = 3.141592653589793 / 2.0;
  int stride = gridDim.x*blockDim.x;
  if(idx>=np)return;
  while(idx<np){
    
    d[idx*nd] += d[idx*nd+1]+d[idx*nd+2];
    d[idx*nd] = sqrt( d[idx*nd]);
    if ( d[idx*nd] < PI2 ){
      d2[idx] = d[idx*nd];
    }else{
      d2[idx] = PI2;
    }
    pe[idx] =  0.5 * pow ( sin ( d2[idx] ), 2 );
    
    idx+=stride;
  }
}
__global__ void compute_f ( int np, int nd,double *d,double *d2,double *f,double *rij,int j){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  int k;
  if(idx>=np*nd)return;
  while(idx<np*nd){

    k = idx / nd;
    if(k!=j){
      f[idx] -=  rij[idx] * sin ( 2.0 * d2[k] ) / d[k*nd];
    }
    // printf("%8f\n",sin ( 2.0 * d2[k] ));
    idx+=stride;
  }
  __syncthreads();
    
}
/********************************************************************************/
__global__  void add_pe(double *pe,int np,int nd,double *OUT,int j){
  __shared__ double sdata[1000];
 int tid = threadIdx.x;
 int i = blockIdx.x*blockDim.x+tid;
 sdata[tid] = 0;
 if(i>=np*nd)return;
 if(gridDim.x!=1){
   while(i<np*nd){
      sdata[tid] += pe[i];
      i += gridDim.x*blockDim.x;
   }
  }else{
   sdata[tid] = pe[tid];
 }
 __syncthreads();
 for(int s=blockDim.x/2;s>0;s/=2){
  if(tid<s){
    sdata[tid]+=sdata[tid+s];
  }
   __syncthreads();
 }
 if(tid==0)pe[blockIdx.x]=sdata[0];
 i = blockIdx.x*blockDim.x+tid;
 if(i==0){
  OUT[j] = pe[0];
 }
}
// compute ke;
/********************************************************************************/
__global__  void add_ke(double *ke,double* vel,int np,int nd, double mass){
  __shared__ double sdata[1000];
 int tid = threadIdx.x;
 int i = blockIdx.x*blockDim.x+tid;
//  printf("%8d\n",i);
  sdata[tid] = 0;
 if(i>=np*nd)return;
 if(gridDim.x!=1){
   while(i<np*nd){
      sdata[tid] += vel[i]*vel[i];
      i += gridDim.x*blockDim.x;
   }
  }else{
   sdata[tid] = ke[tid];
 }
 __syncthreads();
 for(int s=blockDim.x/2;s>0;s/=2){
  if(tid<s){
    sdata[tid]+=sdata[tid+s];
  }
   __syncthreads();
 }
 if(tid==0)ke[blockIdx.x]=sdata[0];
 if(blockIdx.x*blockDim.x+tid==0&&gridDim.x==1)ke[0]*=0.5*mass;
}

/******************************************************************************/
void outputval(double *val,int np,int nd){
  for(int i=0;i<np;i++){
    for(int j=0;j<nd;j++){
      printf("%4f ",val[i*nd+j]);
    }
    printf("\n");
  }
}
int main ( int argc, char *argv[] )
{
  double *acc;
  double *box;
  double dt = 0.0001;
  double e0;
  double *force;
  int i;
  double kinetic;
  double mass = 1.0;
  int nd = 3;
  int np = 1000;
  double *pos;
  double potential;
  int seed = 123456789;
  int step;
  int step_num = 100;
  int step_print;
  int step_print_index;
  int step_print_num;
  double *vel;
  // double wtime;

  timestamp ( );

  acc = ( double * ) malloc ( nd * np * sizeof ( double ) );
  box = ( double * ) malloc ( nd * sizeof ( double ) );
  force = ( double * ) malloc ( nd * np * sizeof ( double ) );
  pos = ( double * ) malloc ( nd * np * sizeof ( double ) );
  vel = ( double * ) malloc ( nd * np * sizeof ( double ) );

  printf ( "\n" );
  printf ( "MD_OPENMP\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "  A molecular dynamics program.\n" );

  printf ( "\n" );
  printf ( "  NP, the number of particles in the simulation is %d\n", np );
  printf ( "  STEP_NUM, the number of time steps, is %d\n", step_num );
  printf ( "  DT, the size of each time step, is %f\n", dt );

  
/*
  Set the dimensions of the box.
*/
  for ( i = 0; i < nd; i++ )
  {
    box[i] = 10.0;
  }

  printf ( "\n" );
  printf ( "  Initializing positions, velocities, and accelerations.\n" );
/*
  Set initial positions, velocities, and accelerations.
*/
  initialize ( np, nd, box, &seed, pos, vel, acc );
/*
  Compute the forces and energies.
*/
  printf ( "\n" );
  printf ( "  Computing initial forces and energies.\n" );
  // memalloc
  double* d_acc, *d_force,*d_pos,*d_vel,*ke,*d,*d2,*pe,*rij,*sumpe;
  cudaMalloc(&d_acc, nd * np * sizeof ( double ));
  cudaMalloc(&d_force, nd * np * sizeof ( double ));
  cudaMalloc(&d_pos, nd * np * sizeof ( double ));
  cudaMalloc(&d_vel, nd * np * sizeof ( double ));
  cudaMalloc(&ke, np *sizeof ( double ));
  cudaMalloc(&rij, nd * np *sizeof ( double ));
  cudaMalloc(&pe, np *sizeof ( double ));
  cudaMalloc(&d, nd * np *sizeof ( double ));
  cudaMalloc(&d2, np *sizeof ( double ));
  cudaMalloc(&sumpe, np *sizeof ( double ));
  // compute sth
  cudaMemset(d_force,0.0,nd * np * sizeof ( double ));
  cudaMemcpy(d_pos, pos, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vel, vel, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);
  cudaMemcpy(d_acc, acc, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);
  // outputval(force,np,nd);
  potential = 0.0;
  // double total_pe = 0.0;
  for(int j=0;j<np;j++){

    compute_rd<< <gridSize, blockSize >> > (np, nd, d_pos, j, d,rij);
    compute_d2 << <gridSize, blockSize >> >(np, nd, d, d2, pe);
    compute_f<< <gridSize, blockSize >> >  (np,nd,d,d2,d_force,rij,j);

    add_pe<< <gridSize, blockSize >> >(pe,1,np,sumpe,j);
    
    // double tmp_pe;
    // cudaMemcpy(&tmp_pe, pe, sizeof ( double ), cudaMemcpyDeviceToHost);
    // total_pe += tmp_pe;
  }
  add_pe<< <gridSize, blockSize >> >(sumpe,1,np,sumpe,0);
  double tmp_pe;
  cudaMemcpy(&tmp_pe, sumpe, sizeof ( double ), cudaMemcpyDeviceToHost);
  printf("%8f\n",tmp_pe);
  potential = tmp_pe;

  add_ke<< <gridSize, blockSize >> >(ke,d_vel,np,nd,mass);
  if(gridSize>1)
    add_ke<< <1, blockSize >> >(ke,ke,1,blockSize,mass);
  double tmp;
  cudaMemcpy(&tmp, ke, sizeof ( double ), cudaMemcpyDeviceToHost);
  kinetic = tmp;
  
  e0 = potential + kinetic;
  
/*
  This is the main time stepping loop:
    Compute forces and energies,
    Update positions, velocities, accelerations.
*/
  printf ( "\n" );
  printf ( "  At each step, we report the potential and kinetic energies.\n" );
  printf ( "  The sum of these energies should be a constant.\n" );
  printf ( "  As an accuracy check, we also print the relative error\n" );
  printf ( "  in the total energy.\n" );
  printf ( "\n" );
  printf ( "      Step      Potential       Kinetic        (P+K-E0)/E0\n" );
  printf ( "                Energy P        Energy K       Relative Energy Error\n" );
  printf ( "\n" );

  step_print = 0;
  step_print_index = 0;
  step_print_num = 10;
  
  step = 0;
  printf ( "  %8d  %14f  %14f  %14e\n",
    step, potential, kinetic, ( potential + kinetic - e0 ) / e0 );
  step_print_index = step_print_index + 1;
  step_print = ( step_print_index * step_num ) / step_print_num;

  StartTimer();;
  // parameter initialization
  
  
  for ( step = 1; step <= step_num; step++ )
  {
    
    cudaMemset(d_force,0.0000000,nd * np * sizeof ( double ));
    // cudaMemcpy(force, d_force, nd * np * sizeof ( double ), cudaMemcpyDeviceToHost);
    
    potential = 0.0;
    for(int j=0;j<np;j++){


      compute_rd<< <gridSize, blockSize >> > (np, nd, d_pos, j, d,rij);
      compute_d2 << <gridSize, blockSize >> >(np, nd, d, d2, pe);
      compute_f<< <gridSize, blockSize >> >  (np,nd,d,d2,d_force,rij,j);

      add_pe<< <gridSize, blockSize >> >(pe,1,np,sumpe,j);
      // tmp_pe<< <    1   ,      1    >> >(sumpe,pe,j);
      // double tmp_pe;
      // cudaMemcpy(&tmp_pe, pe, sizeof ( double ), cudaMemcpyDeviceToHost);
      // total_pe += tmp_pe;

    }
    add_pe<< <gridSize, blockSize >> >(sumpe,1,np,sumpe,0);
    double tmp_pe;
    cudaMemcpy(&tmp_pe, sumpe, sizeof ( double ), cudaMemcpyDeviceToHost);
    // double *f = ( double * ) malloc ( nd * np * sizeof ( double ) );
    //   cudaMemcpy(f, d_force, np*nd*sizeof ( double ), cudaMemcpyDeviceToHost);
    // outputval(f,np,nd);
    potential = tmp_pe;
    
  
    
    
    

    // compute ke
    add_ke<< <gridSize, blockSize >> >(ke,d_vel,np,nd,mass);
    if(gridSize>1)
      add_ke<< <1, blockSize >> >(ke,ke,1,blockSize,mass);
    double tmp;
    cudaMemcpy(&tmp, ke, sizeof ( double ), cudaMemcpyDeviceToHost);
    kinetic = tmp;
    if ( step == step_print )
    {
      printf ( "  %8d  %14f  %14f  %14e\n",
    step, potential, kinetic, ( potential + kinetic - e0 ) / e0 );
      step_print_index = step_print_index + 1;
      step_print = ( step_print_index * step_num ) / step_print_num;
    }
   

    update<< <gridSize, blockSize >> > ( np, nd, d_pos, d_vel, d_force, d_acc, mass, dt );
 
  }
  double runtime = GetTimer();

  printf ( "\n" );
  printf ( "  Elapsed time for main computation:\n" );
  printf ( "  %f seconds.\n", runtime/1000 );
/*
  Free memory.
*/
  free ( acc );
  free ( box );
  free ( force );
  free ( pos );
  free ( vel );
  // cuda
  cudaFree ( d_acc );
  cudaFree ( d_force );
  cudaFree ( d_pos );
  cudaFree ( d_vel );
  cudaFree ( ke );
  cudaFree ( rij );
  cudaFree ( pe );
  cudaFree ( d );
  cudaFree ( d2 );


  
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "MD_OPENMP\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );
  timestamp ( );

  return 0;
}
/******************************************************************************/
/******************************************************************************/

void initialize ( int np, int nd, double box[], int *seed, double pos[], 
  double vel[], double acc[] )

/******************************************************************************/
/*
  Purpose:

    INITIALIZE initializes the positions, velocities, and accelerations.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 November 2007

  Author:

    Original FORTRAN77 version by Bill Magro.
    C version by John Burkardt.

  Parameters:

    Input, int NP, the number of particles.

    Input, int ND, the number of spatial dimensions.

    Input, double BOX[ND], specifies the maximum position
    of particles in each dimension.

    Input, int *SEED, a seed for the random number generator.

    Output, double POS[ND*NP], the position of each particle.

    Output, double VEL[ND*NP], the velocity of each particle.

    Output, double ACC[ND*NP], the acceleration of each particle.
*/
{
  int i;
  int j;
/*
  Give the particles random positions within the box.
*/
  for ( i = 0; i < nd; i++ )
  {
    for ( j = 0; j < np; j++ )
    {
      pos[i+j*nd] = box[i] * r8_uniform_01 ( seed );
    }
  }

  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      vel[i+j*nd] = 0.0;
    }
  }
  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      acc[i+j*nd] = 0.0;
    }
  }
  return;
}
/******************************************************************************/

double r8_uniform_01 ( int *seed )

/******************************************************************************/
/*
  Purpose:

    R8_UNIFORM_01 is a unit pseudorandom R8.

  Discussion:

    This routine implements the recursion

      seed = 16807 * seed mod ( 2**31 - 1 )
      unif = seed / ( 2**31 - 1 )

    The integer arithmetic never requires more than 32 bits,
    including a sign bit.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    11 August 2004

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Springer Verlag, pages 201-202, 1983.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

  Parameters:

    Input/output, int *SEED, a seed for the random number generator.

    Output, double R8_UNIFORM_01, a new pseudorandom variate, strictly between
    0 and 1.
*/
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }

  r = ( double ) ( *seed ) * 4.656612875E-10;

  return r;
}
/******************************************************************************/

void timestamp ( void )
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
/******************************************************************************/
