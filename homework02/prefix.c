#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "common.h"


void usage(int argc, char** argv);
void verify(int* sol, int* ans, int n);
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);


int main(int argc, char** argv)
{
    // get inputs
    uint32_t n = 1048576;
    unsigned int seed = time(NULL);
    if(argc > 2) {
        n = atoi(argv[1]); 
        seed = atoi(argv[2]);
    } else if(argc == 2){
        n = atoi(argv[1]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32" elements and time as seed\n", n);
    }


    // set up data 
    int* prefix_array = (int*) AlignedMalloc(sizeof(int) * n);  
    int* input_array = (int*) AlignedMalloc(sizeof(int) * n);
    srand(seed);
    for(int i = 0; i < n; i++) {
        input_array[i] = rand() % 100;
    }


    // set up timers
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // execute serial prefix sum and use it as ground truth
    start_t = ReadTSC();
    prefix_sum(input_array, prefix_array, n);
    end_t = ReadTSC();
    printf("Time to do O(N-1) prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));


    // execute parallel prefix sum which uses a NlogN algorithm
    int* input_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    int* prefix_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    memcpy(input_array1, input_array, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p1(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do O(NlogN) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);

    
    // execute parallel prefix sum which uses a 2(N-1) algorithm
    memcpy(input_array1, input_array, sizeof(int) * n);
    memset(prefix_array1, 0, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p2(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do 2(N-1) //prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // free memory
    AlignedFree(prefix_array);
    AlignedFree(input_array);
    AlignedFree(input_array1);
    AlignedFree(prefix_array1);


    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stderr, "usage: %s <# elements> <rand seed>\n", argv[0]);
}


void verify(int* sol, int* ans, int n)
{
    int err = 0;
    for(int i = 0; i < n; i++) {
        if(sol[i] != ans[i]) {
            err++;
        }
    }
    if(err != 0) {
        fprintf(stderr, "There was an error: %d\n", err);
    } else {
        fprintf(stdout, "Pass\n");
    }
}

void prefix_sum(int* src, int* prefix, int n)
{
    prefix[0] = src[0];
    for(int i = 1; i < n; i++) {
        prefix[i] = src[i] + prefix[i - 1];
    }
}

void prefix_sum_p1(int* src, int* prefix, int n)
{
    // We will traverse the source array from left to right over and over
    // increasing the distance we cover in each step AFTER each full array traversal 
    // by a power of 2

    int* og_prefix = prefix;

    for (int stepper = 1; stepper < n; stepper *= 2){
        // inside the loop we will tell a core to add elements from the left

        #pragma omp parallel for
        for (int i = 0; i < n; i++){
            // if we find an element at i - stepper we add i and i - stepper
            // to avoid waiting for variables to be written to while we read them
            // manage two lists, one to read from and one to write to. 
            
            if (i >= stepper){

                prefix[i] = src[i] + src[i - stepper];
            // after the first loop we have built prefix to be the a partial sum
            // such that each next element is a sum of itself and the element
            // to its left in the source
            } // if we dont find an element to the left, simply copy previous solution. 


            else{

                // since its either the first element of the list or its a complete sum (no longer partial)

                prefix[i] = src[i];

            }
        }	
        // swap the arrays to build off of what we just updated 

        int* temp = src;
        src = prefix;
        prefix = temp;
    }

    // depending on the number of steps (size of n) we either end with the last sum 
    // correctly added to prefix or correctly added to src. If the final result is in 
    // source, we must copy the data to the location we expect to read from for our comparison 
    // after the function is called. Because we swap pointers of the arrays, saving a pointer
    // to the original prefix array allows us to make sure we save data to the right place before 
    // returning.

    if (src != og_prefix){
        // we copied the final result to the wrong comparitor
        # pragma omp parallel for
        for (int i = 0; i < n; i++){

            // copy complete sum to the final array
            og_prefix[i] = src[i];
        }        
    }
}

void prefix_sum_p2(int* src, int* prefix, int n)
{
    /* instead of adding elements, we add spans of partial sums
    in order to organize the partial sums, we create a tree

    I ran into an issue with arrays that were not exact powers of 2
    with my original initialization of the tree. 

    int* tree = malloc(n * sizeof(int));
   
    We must pad the array to the next power to align with the way 
    the source and prefix arrays are parsed lower in function.
    */
    
    int next_power = 1;
    // to reach our next power of 2 value
    while (next_power < n) next_power*=2;
    int* tree = malloc(next_power * sizeof(int));
    // populate it with initial values
    
    # pragma omp parallel for 
    for (int i = 0; i < next_power; i++){
        if (i < n){
            tree[i] = src[i];
        } else {
            tree[i] = 0;
        }
    }
    // like the last, we will double our stepper at each iteration
    // unlike our last implementation we need not iterate over every node
    // in our first pass. In this way, we divide the list in half each pass.
    for (int stepper = 2; stepper <= next_power; stepper*=2){
        # pragma omp parallel for
        for (int i = stepper - 1; i < next_power; i += stepper){
            // Add left and right subtree (2 children of internal node)    
            tree[i] += tree[i - stepper / 2]; 
        } 
    }

    // Now tree[next_power-1] contains the sum of all elements 
    tree[next_power - 1] = 0;

    // Now we reverse our traversal to add the partials 
    for (int stepper = next_power; stepper >= 2; stepper /= 2){
        
        # pragma omp parallel for
        for (int i = stepper - 1; i < next_power; i += stepper){
            // Save the left subtree 
            int temp = tree[i - stepper / 2];
            // The left child inherits the parent's prefix sum (starts as 0)
            tree[i - stepper / 2] = tree[i];

            // The right child inherits the parent's prefix plus
            // the left subtree's partial sum
            tree[i] += temp;
        } 
    }
    # pragma omp parallel for
    for (int i = 0; i < n; i++){
        prefix[i] = tree[i] + src[i];
    }
    free(tree);
}



