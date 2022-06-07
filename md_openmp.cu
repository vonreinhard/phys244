#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"

int main ( int argc, char *argv[] );
void compute ( int np, int nd, double pos[], double vel[], 
  double mass, double f[], double *pot, double *kin );
double dist ( int nd, double r1[], double r2[], double dr[] );
void initialize ( int np, int nd, double box[], int *seed, double pos[], 
  double vel[], double acc[] );
double r8_uniform_01 ( int *seed );
void timestamp ( );

#define gridSize 2
#define blockSize 100
__global__ void update ( int np, int nd, double* pos, double* vel, double* f, double* acc, double mass, double dt )
{
 
  double rmass;

  rmass = 1.0 / mass;



  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("%8d\n",idx);
  if(idx < (np*nd) ){

    pos[idx] = pos[idx] + vel[idx] * dt + 0.5 * acc[idx] * dt * dt;
    vel[idx] = vel[idx] + 0.5 * dt * ( f[idx] * rmass + acc[idx] );
    acc[idx] = f[idx] * rmass;
  }

  return;
}
/*********************************/
__global__ void set_f(double* f,int np,int nd){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<np*nd){
    f[idx] = 0.0;
  }
}
__global__ void compute_cu ( int np, int nd, double* pos, double* f ){
    double d=0;
    double d2;
    
    double pe;
    double PI2 = 3.141592653589793 / 2.0;
    double rij[3];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = idx/np;
    int i = idx%np;
    if(idx<np*nd){
      
      for(int j=0;j<np;j++){
        if(j==k)continue;
        //compute d
     

       
        
        
        
        d = sqrt ( d );

        if ( d < PI2 ){
          d2 = d;
        }else{
          d2 = PI2;
        }

        pe = pe + 0.5 * pow ( sin ( d2 ), 2 );
        f[idx] = f[idx]- rij[i] * sin ( 2.0 * d2 ) / d;
      }

    }
  }
  
// compute ke;
/********************************************************************************/
__global__  void add_ke(double *ke,double* vel,int np,int nd, double mass){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    double sum = 0.0;
    if(idx>=np*nd)return;
    while(idx<np*nd){
      sum+= vel[idx]*vel[idx];
      
      
      idx +=stride;
    }

    int thIdx = threadIdx.x;
    __shared__ float shArr[blockSize];
    if(gridDim.x!=1)
      shArr[thIdx] =sum;
    else
      shArr[thIdx] = ke[thIdx];
    __syncthreads();
    for(int size = blockSize/2;size>0;size/=2){
    	if(thIdx<size)
		    shArr[thIdx]+= shArr[thIdx+size];
	    __syncthreads();
    }
    
    ke[blockIdx.x] = shArr[0];

    if(gridDim.x==1&&thIdx ==0){
      
      ke[0] *= 0.5*mass;
      // printf("%14f %8d %8d,%8d\n",ke[0],gridDim.x,blockDim.x,blockIdx.x);
    }
 
}
/******************************************************************************/

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
  int np = 10;
  double *pos;
  double potential;
  int seed = 123456789;
  int step;
  int step_num = 100;
  int step_print;
  int step_print_index;
  int step_print_num;
  double *vel;
  double wtime;

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

  compute ( np, nd, pos, vel, mass, force, &potential, &kinetic );

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
  double* d_acc, *d_force,*d_pos,*d_vel,*d_kinetic,*ke;
  cudaMalloc(&d_acc, nd * np * sizeof ( double ));
  cudaMalloc(&d_force, nd * np * sizeof ( double ));
  cudaMalloc(&d_pos, nd * np * sizeof ( double ));
  cudaMalloc(&d_vel, nd * np * sizeof ( double ));
  cudaMalloc(&ke, blockSize *sizeof ( double ));
  d_kinetic = ( double * ) malloc (sizeof ( double ) );
  // cudaMalloc(&d_kinetic, sizeof ( double ));

  for ( step = 1; step <= step_num; step++ )
  {
    cudaMemcpy(d_pos, pos, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_force, force, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acc, acc, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);

    set_f<< <gridSize, blockSize >> > (d_force,np,nd);
    // compute_cu<< <gridSize, blockSize >> > ( np, nd, d_pos, d_force);
    compute ( np, nd, pos, vel, mass, force, &potential, &kinetic );
    // compute ke
    // printf("ttt\n");
    add_ke<< <gridSize, blockSize >> >(ke,d_vel,np,nd,mass);
    // if(gridSize>1)
    //   add_ke<< <1, blockSize >> >(ke,ke,1,blockSize,mass);
    double *tmp = ( double * ) malloc (blockSize*sizeof ( double ) );
    cudaMemcpy(tmp, ke, blockSize*sizeof ( double ), cudaMemcpyDeviceToHost);
    // cudaMemcpy(d_kinetic, ke, sizeof ( double ), cudaMemcpyDeviceToHost);
    double sum = 0;
    for(int i=0;i<blockSize;i++){
      sum+=tmp[i];
    }
    *d_kinetic = sum*0.5*mass;
    if(*d_kinetic!=kinetic){
      printf("%14f %14f \n",*d_kinetic,kinetic);
    }
    if ( step == step_print )
    {
      printf ( "  %8d  %14f  %14f  %14e\n", step, potential, kinetic,
       ( potential + kinetic - e0 ) / e0 );
      step_print_index = step_print_index + 1;
      step_print = ( step_print_index * step_num ) / step_print_num;
    }
   
    cudaMemcpy(d_pos, pos, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_force, force, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acc, acc, nd * np * sizeof ( double ), cudaMemcpyHostToDevice);

    update<< <gridSize, blockSize >> > ( np, nd, d_pos, d_vel, d_force, d_acc, mass, dt );

    cudaMemcpy(pos, d_pos, nd * np * sizeof ( double ), cudaMemcpyDeviceToHost);
    cudaMemcpy(vel, d_vel, nd * np * sizeof ( double ), cudaMemcpyDeviceToHost);
    cudaMemcpy(force, d_force, nd * np * sizeof ( double ), cudaMemcpyDeviceToHost);
    cudaMemcpy(acc, d_acc, nd * np * sizeof ( double ), cudaMemcpyDeviceToHost);
    // printf ( "%14f\n",pos[0] );
  }
  //wtime = GetTimer() ;

  printf ( "\n" );
  printf ( "  Elapsed time for main computation:\n" );
  //printf ( "  %f seconds.\n", wtime );
/*
  Free memory.
*/
  free ( acc );
  free ( box );
  free ( force );
  free ( pos );
  free ( vel );
  free ( d_kinetic );
  // cuda
  cudaFree ( d_acc );
  cudaFree ( d_force );
  cudaFree ( d_pos );
  cudaFree ( d_vel );
  cudaFree ( ke );
  
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

void compute ( int np, int nd, double pos[], double vel[], 
  double mass, double f[], double *pot, double *kin )
{
  double d;
  double d2;
  int i;
  int j;
  int k;
  double ke;
  double pe;
  double PI2 = 3.141592653589793 / 2.0;
  double rij[3];

  pe = 0.0;
  ke = 0.0;

# pragma omp parallel \
  shared ( f, nd, np, pos, vel ) \
  private ( i, j, k, rij, d, d2 )
  

# pragma omp for reduction ( + : pe, ke )
  for ( k = 0; k < np; k++ )
  {
/*
  Compute the potential energy and forces.
*/
    for ( i = 0; i < nd; i++ )
    {
      f[i+k*nd] = 0.0;
    }

    for ( j = 0; j < np; j++ )
    {
      if ( k != j )
      {
        d = dist ( nd, pos+k*nd, pos+j*nd, rij );
        
/*  
  Attribute half of the potential energy to particle J.
*/
        if ( d < PI2 )
        {
          d2 = d;
        }
        else
        {
          d2 = PI2;
        }

        pe = pe + 0.5 * pow ( sin ( d2 ), 2 );

        for ( i = 0; i < nd; i++ )
        {
          f[i+k*nd] = f[i+k*nd] - rij[i] * sin ( 2.0 * d2 ) / d;
        }
      }
    }
/*
  Compute the kinetic energy.
*/
    for ( i = 0; i < nd; i++ )
    {
      ke = ke + vel[i+k*nd] * vel[i+k*nd];
    }
  }

  ke = ke * 0.5 * mass;
  
  *pot = pe;
  *kin = ke;

  return;
}
/******************************************************************************/

double dist ( int nd, double r1[], double r2[], double dr[] )
{
  double d;
  int i;

  d = 0.0;
  for ( i = 0; i < nd; i++ )
  {
    dr[i] = r1[i] - r2[i];
    d = d + dr[i] * dr[i];
  }
  d = sqrt ( d );

  return d;
}
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

