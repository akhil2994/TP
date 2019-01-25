/*  B145208  */

#include <stdio.h>
#include <math.h>

#define N 729
#define reps 1000
#include <omp.h>

double a[N][N], b[N][N], c[N];
int jmax[N];

void init1(void);
void init2(void);
void runloop(int);
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);

int main(int argc, char *argv[]) {
    double start1, start2, end1, end2;
    int r;

    init1();

    start1 = omp_get_wtime();

    for (r=0; r<reps; ++r) {
        runloop(1);
    }

    end1  = omp_get_wtime();

    valid1();

    printf("Total time for %d reps of loop 1 = %f\n",reps, (float)(end1-start1));

    init2();

    start2 = omp_get_wtime();

    for (r=0; r<reps; ++r) {
        runloop(2);
    }

    end2  = omp_get_wtime();

    valid2();

    printf("Total time for %d reps of loop 2 = %f\n",reps, (float)(end2-start2));
}

void init1(void) {
    int i, j;

    for (i=0; i<N; ++i) {
        for (j=0; j<N; ++j) {
            a[i][j] = 0.0;
            b[i][j] = 3.142*(i+j);
        }
    }
}

void init2(void) {
    int i,j, expr;

    for (i=0; i<N; ++i) {
        expr =  i%( 3*(i/30) + 1);

        if ( expr == 0) {
            jmax[i] = N;
        } else {
            jmax[i] = 1;
        }

        c[i] = 0.0;
    }

    for (i=0; i<N; ++i) {
        for (j=0; j<N; ++j) {
            b[i][j] = (double) (i*j+1) / (double) (N*N);
        }
    }
}

void runloop(int loopid) {

    int max_threads = omp_get_max_threads();
    int remain_iteration[max_threads];
    int thread_array[max_threads]; //this will be used to find the thread which has not completed any of the chunks assigned

    #pragma omp parallel default(none) shared(loopid, remain_iteration, thread_array)
    {
        int myid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int ipt = (int) ceil((double)N/(double)nthreads);
        int lo = myid*ipt; //calculation low for every thread
        int hi = (myid+1)*ipt; // calculating high for every thread

        if (hi > N) hi = N;

        remain_iteration[myid] = hi - lo;
        thread_array[myid] = 0;

        /* This barrier directive will ensure two things.
         * First to make sure all threads reaches this position prior to actual calculation
         * Second - every thread has its own array values (reamin_iteration and thread_array)
        */
        #pragma omp barrier
        int index = 1;
        int track_thread;
        
        /* This section of code contains if-else condition
         * if condition is used for threads to execute their own chunks
         * else is used to find the thread with maximum iterations left
        */
        while (index) {
            #pragma omp critical
            {
                if (remain_iteration[myid] > 0) {
                    index = 1;
                    track_thread = myid;
                }
                else {
                    int i, most_loaded_thread_value;
                    most_loaded_thread_value = 0;
                    
                    for (i=0; i<nthreads; ++i) {
                        if (thread_array[i] == 0 && remain_iteration[i] > most_loaded_thread_value) {
                            track_thread = i; 
                            most_loaded_thread_value = remain_iteration[i];
                        }
                    }
                    //checking if most loaded has iterations left
                    index = most_loaded_thread_value > 0 ? 1 : 0 ;
                }
            } //end of first critical region


            if (index) {
                int one_p, local_lo, local_hi;

                #pragma omp critical
                {
                    /* This section of code deals with creating the Guided Scheduling
                     * Calculates the local chunks
                     * Calculates the 1/p value for local chunk execution
                    */
                    int iterations_with_thread = (track_thread + 1) * ipt;
                    if (iterations_with_thread > N) iterations_with_thread = N;
                    
                    one_p = (int) ceil((1.0 / (double)nthreads) * (double)remain_iteration[track_thread]);
                    local_lo = iterations_with_thread - remain_iteration[track_thread]; //subtracting the remaining iterations of threads from maximum iterations available with it. Example 183-137, in case of thread 0 with 2nd local chunk
                    local_hi = local_lo + one_p;
                    
                    /* To make sure shared is updated
                     * Every thread process the shared array
                     * To avoid race conditions
                    */
                    thread_array[track_thread] = 1;
                    remain_iteration[track_thread] -= one_p;
                }

                
                switch (loopid) {
                case 1:
                    loop1chunk(local_lo, local_hi);
                    break;
                case 2:
                    loop2chunk(local_lo, local_hi);
                    break;
                }
                #pragma omp critical
                {
                    // Finally, set thread flag back to idle.
                    thread_array[track_thread] = 0;
                }
            }
        }
    }
}

void loop1chunk(int lo, int hi) {
    int i, j;

    for (i=lo; i<hi; ++i) {
        for (j=N-1; j>i; --j) {
            a[i][j] += cos(b[i][j]);
        }
    }
}

void loop2chunk(int lo, int hi) {
    int i, j, k;
    double rN2;

    rN2 = 1.0 / (double) (N*N);

    for (i=lo; i<hi; ++i) {
        for (j=0; j < jmax[i]; ++j) {
            for (k=0; k<j; ++k) {
                c[i] += (k+1) * log (b[i][j]) * rN2;
            }
        }
    }
}

void valid1(void) {
    int i, j;
    double suma;

    suma = 0.0;
    for (i=0; i<N; ++i) {
        for (j=0; j<N; ++j) {
            suma += a[i][j];
        }
    }

    printf("Loop 1 check: Sum of a is %lf\n", suma);
}

void valid2(void) {
    int i;
    double sumc;

    sumc = 0.0;
    for (i=0; i<N; ++i) {
        sumc += c[i];
    }

    printf("Loop 2 check: Sum of c is %f\n", sumc);
}

