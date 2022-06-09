# <center>Final Report</center>
<center>TzuKao Wang (A59010714)</center>
<center>Jack Sun (A16062902)</center>
<center>Zhengtong Zhang (A59011254)</center>

## Section 1 Introduction
Molecular dynamics is a kind of simulation method to analyze atoms and
molecules’s physical movements.When given the original position and the velocity of
the particles, we could use many-body Newton functions to predict the particles
movement in the future and hence simulate the whole process of the system.

Since we could consider the system would become an ergodic system in the long term, the computation of each particles’ movement could be processed in the parallel approach. Thus we use OpenMP as the parallel computation method to assist the simulation of molecular dynamics. We will test the parallel program and develop the performance model based on the parameter of processors, the number of particles and the process time from computation and communcation. For further comparison, we will also utilizz OpenMC and Cuda method as other parallel technique for this program. 

## Section 2 Technique Appraoch

### 2.1 OpenMp

### 2.2 OpenACC

### 2.3 CUDA

## Section 3 Developing Process
We will describe the developing process for each parallelize method, including the chanllenges and the notion we made during developing.
### 3.1 OpenMP

### 3.2 OpenACC
Since OpenAcc is much similar with OpenMP, its developing process could refer from the OpenMp version. For the $\mathbf{update}$, it seems that we could directly parallel them withm three matrixs copy and force matrix just need to be sent to device without sending back. $\mathbf{compute}$ function , the situation is more complex but we still could handle it. When we look at potential and kinetic, we are doing the combination. Thus, we could use the reduction. Furthermore, for other computation we found the rij(distance for different particle) and degree could be used privately. Thus. after implementing those method, we success finished the OpenAcc version.
### 3.3 CUDA
From my observation, there are two part we need to do the parallelization which first part is $\mathbf{compute}$ function and the other is $\mathbf{update}$ part.

The later one is much easy since there is not data dependecy in this function and thus we could assign the thread to complete their tasks without waiting and communication with other threads.

But the first part is not such stright forward. There are three level for loops and many sub for loops in this function. It is hard for us to spilt to each thread. But we know that in many situation, the best approach to achieve serial method may not be also a good method when doing the parallelization. I observer that we could reorganized function and then it will be easy parallelization for CUDA method. I would use md_openmp.c document as the reference to describe how I reorganize the function in the following

At first, we could observer that the initialization from line 269-272 will occur once for each element in force array. The program from 302-305 has similar situation. For initialization, we could use $\mathbf{cudamemset}$ before this program starts. For the later part, the array $\mathbf{vel}$ and the value $\mathbf{ke}$ has no depency with above function. Thus, we could divide then into two different part one is doing above and the other does the ke computation. The parallelization for above part is still hard, but is much easier now.

In the remaining unparallelized program, we found that we could swith k and j now. After switching, we could do the parallelization under the for loop of j and we just need to do an extra reduction add to sum the result in this loop and then it could be paralleliazation.

## Section 4 Performance Model

### 4.1 OpenMP

### 4.2 OpenACC

### 4.3 CUDA

## Section 5 Conclusion




## Section 6 Future work 
For CUDA, we may use some method to parallelized the j loop. But this need more syncronization problem. Besides, the reduction could be optimized more. BUt due to time limitation, we would left it in the next step for our project.

For OpenAcc version, we will try to reduce the action for sent the data from fots to device and back. We believe it could optimized our result in the future.

## Reference
[1] John burkardt. (n.d.). Md_openmp.
https://people.sc.fsu.edu/~jburkardt/cpp_src/md_openmp