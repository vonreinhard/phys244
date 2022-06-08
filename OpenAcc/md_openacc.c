# include <math.h>
#include <string.h>
#include <openacc.h>
# include <stdlib.h>
# include <stdio.h>
#include "timer.h"
#include <time.h>


int main ( int argc, char *argv[] );
void compute ( int np, int nd, float pos[], float vel[], 
  float mass, float f[], float *pot, float *kin );
// float dist ( int nd, float r1[], float r2[], float dr[] );
void initialize ( int np, int nd, float box[], int *seed, float pos[], 
  float vel[], float acc[] );
float r8_uniform_01 ( int *seed );
void timestamp ( );
void update ( int np, int nd, float pos[], float vel[], float f[], 
  float acc[], float mass, float dt );


int main ( int argc, char *argv[] )
{
  // float *acc;
  // float *box;
  // float *force;
  // float *pos;
  // float *vel;

  float dt = 0.0001;
  float e0;
  int i;
  float kinetic;
  float mass = 1.0;
  int nd = 3;
  int np = 1000;
  float potential;
  int seed = 123456789;
  int step;
  int step_num = 400;
  int step_print;
  int step_print_index;
  int step_print_num;
  float wtime;

  timestamp ( );

  float *box = ( float * ) malloc ( nd * sizeof ( float ) );

  float *restrict acc;
  float *restrict force;
  float *restrict pos;
  float *restrict vel;

  acc = ( float * ) malloc ( nd * np * sizeof ( float ) );
  force = ( float * ) malloc ( nd * np * sizeof ( float ) );
  pos = ( float * ) malloc ( nd * np * sizeof ( float ) );
  vel = ( float * ) malloc ( nd * np * sizeof ( float ) );

  printf ( "\n" );
  printf ( "MD_OPENACC\n" );
  printf ( "  C/OpenACC version\n" );
  printf ( "  A molecular dynamics program.\n" );

  printf ( "\n" );
  printf ( "  NP, the number of particles in the simulation is %d\n", np );
  printf ( "  STEP_NUM, the number of time steps, is %d\n", step_num );
  printf ( "  DT, the size of each time step, is %f\n", dt );

//   printf ( "\n" );
//   printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
//   printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
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

  // #pragma acc data copyin(force[:nd+(np*nd)], acc[:nd+(np*nd)], pos[:nd+(np*nd)], vel[:nd+(np*nd)], dt, nd, np, mass)

  wtime = GetTimer();

  for ( step = 1; step <= step_num; step++ )
  {
    compute ( np, nd, pos, vel, mass, force, &potential, &kinetic );

    if ( step == step_print )
    {
      printf ( "  %8d  %14f  %14f  %14e\n", step, potential, kinetic,
       ( potential + kinetic - e0 ) / e0 );
      step_print_index = step_print_index + 1;
      step_print = ( step_print_index * step_num ) / step_print_num;
    }
    update ( np, nd, pos, vel, force, acc, mass, dt );
  }

  wtime =GetTimer() - wtime;

  printf ( "\n" );
  printf ( "  Elapsed time for main computation:\n" );
  printf ( "  %f seconds.\n", wtime );
/*
  Free memory.
*/
  free ( acc );
  free ( box );
  free ( force );
  free ( pos );
  free ( vel );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "MD_OPENACC\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );
  timestamp ( );

  return 0;
}

/******************************************************************************/

void compute ( int np, int nd, float pos[], float vel[], 
  float mass, float f[], float *pot, float *kin )
{
  // float d;
  // float d2;
  // int i;
  // int j;
  // int k;
  float rij[3];
  float ke;
  float pe;
  // float PI2 = 3.141592653589793 / 2.0;

  pe = 0.0;
  ke = 0.0;

// # pragma omp parallel \
//   shared ( f, nd, np, pos, vel ) \
//   private ( i, j, k, rij, d, d2 )
// # pragma omp for reduction ( + : pe, ke )

#pragma acc data copyin(f[:nd+(np*nd)], pos[:nd], vel[:nd], nd, np) 
#pragma acc data copy(pe, ke)

// #pragma acc update device(rij[0:nd])
// #pragma acc kernels 
// #pragma acc region 
#pragma acc parallel loop gang vector private(rij)
  for ( int k = 0; k < np; k++ )
  {
/*
  Compute the potential energy and forces.
*/
    for ( int i = 0; i < nd; i++ )
    {
      f[i+k*nd] = 0.0;
    }

    #pragma acc loop reduction(+ : pe) 
      for ( int j = 0; j < np; j++ )
      {
        if ( k != j )
        {
          // float rij[3]; 
          // d = dist ( nd, pos+k*nd, pos+j*nd, rij );

          float d = 0.0;
          #pragma acc loop
            for ( int i = 0; i < nd; i++ )
            {
              rij[i] = (pos+k*nd)[i] - (pos+j*nd)[i];
              d = d + rij[i] * rij[i];
            }
          d = sqrt ( d );
  /*
    Attribute half of the potential energy to particle J.
  */      
          float d2;
          float PI2 = 3.141592653589793 / 2.0;
          if ( d < PI2 )
          {
            d2 = d;
          }
          else
          {
            d2 = PI2;
          }

          pe = pe + 0.5 * pow ( sin ( d2 ), 2 );

          #pragma acc loop 
            for ( int i = 0; i < nd; i++ )
            {
              f[i+k*nd] = f[i+k*nd] - rij[i] * sin ( 2.0 * d2 ) / d;
            }
        }
      }
/*
  Compute the kinetic energy.
*/
    #pragma acc loop reduction(+ : ke)
      for ( int i = 0; i < nd; i++ )
      {
        ke = ke + vel[i+k*nd] * vel[i+k*nd];
      }
  }

  // #pragma acc update host(pe, ke)

  ke = ke * 0.5 * mass;
  
  *pot = pe;
  *kin = ke;

  return;
}

/******************************************************************************/

// float dist ( int nd, float r1[], float r2[], float dr[] )
// {
//   float d;
//   int i;

//   d = 0.0;
//   // #pragma acc kernels
//     for ( i = 0; i < nd; i++ )
//     {
//       dr[i] = r1[i] - r2[i];
//       d = d + dr[i] * dr[i];
//     }
//   d = sqrt ( d );

//   return d;
// }

/******************************************************************************/

void initialize ( int np, int nd, float box[], int *seed, float pos[], 
  float vel[], float acc[] )
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

float r8_uniform_01 ( int *seed )
{
  int k;
  float r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }

  r = ( float ) ( *seed ) * 4.656612875E-10;

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

void update ( int np, int nd, float pos[], float vel[], float f[], 
  float acc[], float mass, float dt )
{
  // int i;
  // int j;
  float rmass = 1.0 / mass;

// # pragma omp parallel \
//   shared ( acc, dt, f, nd, np, pos, rmass, vel ) \
//   private ( i, j )
// # pragma omp for

#pragma acc data copyin(f[:nd+(np*nd)], acc[:nd+(np*nd)], pos[:nd+(np*nd)], vel[:nd+(np*nd)], dt, nd, np, rmass)

// #pragma acc kernels
// #pragma acc region
// #pragma acc loop independent vector(16)

#pragma acc parallel loop
  for (int j = 0; j < np; j++ )
  {
    #pragma acc loop
    for (int i = 0; i < nd; i++ )
    {
      pos[i+j*nd] = pos[i+j*nd] + vel[i+j*nd] * dt + 0.5 * acc[i+j*nd] * dt * dt;
      vel[i+j*nd] = vel[i+j*nd] + 0.5 * dt * ( f[i+j*nd] * rmass + acc[i+j*nd] );
      acc[i+j*nd] = f[i+j*nd] * rmass;
    }
  }

  return;
}
