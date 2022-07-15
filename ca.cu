/* Cellularautomaton, paralell implementation with CUDA
 * Hannah Peuckmann, Matr.-Nr.:791996, WiSe 2021/22
 * #1: Number of lines
 * #2: Number of iterations to be simulated
 *
 */
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "openssl/md5.h"
#include "random.h"
#include <fcntl.h>


/* horizontal size of the configuration */
#define XSIZE 1024

/* "ADT" State and line of states (plus border) */
typedef char State;
typedef State Line[XSIZE + 2];

// Error checking and time measurement setup
#define CUDA_ERR_CHECK(x) do { cudaError_t err = x; if (( err ) != cudaSuccess ) { \
  printf ("Error \"%s\" at %s :%d \n" , cudaGetErrorString(err), \
        __FILE__ , __LINE__ ) ; exit(-1); \
}} while (0)

#define TIME_GET(timer) \
  struct timespec timer; \
  clock_gettime(CLOCK_MONOTONIC, &timer)

#define TIME_DIFF(timer1, timer2) \
  ((timer2.tv_sec * 1.0E+9 + timer2.tv_nsec) - \
  (timer1.tv_sec * 1.0E+9 + timer1.tv_nsec)) / 1.0E+9

/* determine random integer between 0 and n-1 */
#define randInt(n) ((int)(nextRandomLEcuyer() * n))

/* get MD5 checksum string of a memory chunk */
char* getMD5DigestStr(void* buf, size_t buflen)
{
  MD5_CTX ctx;
	unsigned char sum[MD5_DIGEST_LENGTH];
	int i;
	char* retval;
	char* ptr;

	MD5_Init(&ctx);
	MD5_Update(&ctx, buf, buflen);
	MD5_Final(sum, &ctx);

	retval = (char*)calloc(MD5_DIGEST_LENGTH * 2 + 1, sizeof(*retval));
	ptr = retval;

	for (i = 0; i < MD5_DIGEST_LENGTH; i++) {
		snprintf(ptr, 3, "%02X", sum[i]);
		ptr += 2;
	}

	return retval;
}


/* --------------------- CA simulation -------------------------------- */

/* random starting configuration */
static void initConfig(Line *buf, int lines)
{
	int x, y;

	initRandomLEcuyer(424243);
	for (y = 1;  y <= lines;  y++) {
		for (x = 1;  x <= XSIZE;  x++) {
			buf[y][x] = randInt(100) >= 50;
		}
	}
}


/* treat torus like boundary conditions */
__global__ static void boundary(Line *buf, int lines)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (y == lines-1) {
    buf[y+2][x+1] = buf[1][x+1];
  }
  if (y == 0) {
    buf[y][x+1] = buf[lines][x+1];
  }
  // We only have 'lines' times threads on the y axis so the edges are not copied over
  if (x == 0) {
    buf[y+1][x] = buf[y+1][XSIZE];
  }
  if (x == XSIZE-1) {
    buf[y+1][x+2] = buf[y+1][1];
  }

  // Extra step to copy over the four edges, make sure all the steps above are finished to ensure that there is data to copy from
  if (x == 0 && y == 0){
    buf[0][0] = buf[lines][XSIZE];
    buf[0][XSIZE+1] = buf[lines][1];
    buf[lines+1][0] = buf[1][XSIZE];
    buf[lines+1][XSIZE+1] = buf[1][1];

  }

}

/* make one simulation iteration with 'lines' lines.
* old configuration is in from, new one is written to 'to'.
*/
__global__ void simulate(Line *from, Line *to, int lines)
{

  static State anneal[10] = {0, 0, 0, 0, 1, 0, 1, 1, 1, 1};
  int y = (blockIdx.y * blockDim.y + threadIdx.y) +1;
  int x = (blockIdx.x * blockDim.x + threadIdx.x) +1;
  to[y][x]= anneal[(from)[(y)-1][(x)-1] + (from)[(y)][(x)-1] + (from)[(y)+1][(x)-1] +\
          (from)[(y)-1][(x)  ] + (from)[(y)][(x)  ] + (from)[(y)+1][(x)  ] +\
          (from)[(y)-1][(x)+1] + (from)[(y)][(x)+1] + (from)[(y)+1][(x)+1]];
}


/* --------------------- measurement ---------------------------------- */

int main(int argc, char** argv)
{
  int lines, iterations;
  Line *pFrom, *pTo, *pTemp; // device
  char* hash;

  assert(argc == 3);

  lines = atoi(argv[1]);
  iterations = atoi(argv[2]);

  from = (Line*)calloc((lines + 2), sizeof(Line));
  to   = (Line*)calloc((lines + 2), sizeof(Line));

  initConfig(from, lines);

  if(from == NULL | to == NULL){
    printf("ERROR: failure allocating host memory\n");
    exit(EXIT_FAILURE);
  }

  // allocate device memory
  cudaMalloc((void **)&pFrom, (lines +2) * sizeof(Line));
  cudaMalloc((void **)&pTo, (lines +2) * sizeof(Line));

  if(pFrom == NULL | pTo == NULL){
    printf("ERROR: failure allocating device memory\n");
    exit(EXIT_FAILURE);
  }

  TIME_GET(start);
  dim3 numBlocks(32, 4096);
  dim3 threadsPerBlock(32, 32); // max threads per block = 1024

  cudaMemcpy(pFrom, from, (lines +2) * sizeof(Line), cudaMemcpyHostToDevice);
  for (int i= 0; i < iterations; i++) {
    boundary<<<numBlocks, threadsPerBlock>>>(pFrom, lines);
    simulate<<<numBlocks, threadsPerBlock>>>(pFrom, pTo, lines);
    pTemp = pFrom;
    pFrom = pTo;
    pTo = pTemp;
  }
  cudaMemcpy(from, pFrom, (lines+2) * sizeof(Line), cudaMemcpyDeviceToHost);
  CUDA_ERR_CHECK( cudaGetLastError() );
  TIME_GET(end);

	hash = getMD5DigestStr(from,lines);
	printf("hash gpu: %s\ttime: %.1f ms\n", hash, TIME_DIFF(start,end)*1000);

  cudaFree(pFrom);
  cudaFree(pTo);
  cudaFree(pTemp);

  free(from);
  free(to);
  free(hash);
  return EXIT_SUCCESS;
}