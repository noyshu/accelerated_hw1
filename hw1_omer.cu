/* compile with: nvcc -O3 hw1.cu -o hw1 */

#include <stdio.h>
#include <sys/time.h>

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define IMAGE_SIZE 1024

typedef unsigned char uchar;
#define OUT

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

/* we won't load actual files. just fill the images with random bytes */
void load_image_pairs(uchar *images1, uchar *images2) {
    srand(0);
    for (int i = 0; i < N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION; i++) {
        images1[i] = rand() % 256;
        images2[i] = rand() % 256;
    }
}

__host__ __device__ bool is_in_image_bounds(int i, int j) {
    return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
}

__host__ __device__ uchar local_binary_pattern(uchar *image, int i, int j) {
    uchar center = image[i * IMG_DIMENSION + j];
    uchar pattern = 0;
    if (is_in_image_bounds(i - 1, j - 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
    if (is_in_image_bounds(i - 1, j    )) pattern |= (image[(i - 1) * IMG_DIMENSION + (j    )] >= center) << 6;
    if (is_in_image_bounds(i - 1, j + 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
    if (is_in_image_bounds(i    , j + 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j + 1)] >= center) << 4;
    if (is_in_image_bounds(i + 1, j + 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
    if (is_in_image_bounds(i + 1, j    )) pattern |= (image[(i + 1) * IMG_DIMENSION + (j    )] >= center) << 2;
    if (is_in_image_bounds(i + 1, j - 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
    if (is_in_image_bounds(i    , j - 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j - 1)] >= center) << 0;
    return pattern;
}

//__device__ void zero

void image_to_histogram(uchar *image, int *histogram) {
    memset(histogram, 0, sizeof(int) * 256);
    for (int i = 0; i < IMG_DIMENSION; i++) {
        for (int j = 0; j < IMG_DIMENSION; j++) {
            uchar pattern = local_binary_pattern(image, i, j);
            histogram[pattern]++;
        }
    }
}

double histogram_distance(int *h1, int *h2) {
    /* we'll use the chi-square distance */
    double distance = 0;
    for (int i = 0; i < 256; i++) {
        if (h1[i] + h2[i] != 0) {
            distance += ((double)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
        }
    }
    return distance;
}

/* Your __device__ functions and __global__ kernels here */
/* ... */
__global__ void image_to_hisogram_simple(uchar *image1, OUT int *hist1) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    uchar pattern = local_binary_pattern(image1, i, j);
    atomicAdd(hist1+pattern,1);
   // __threadfence();
}
__global__ void histogram_distance(int *hist1, int *hist2, OUT double *distance) {
    *distance=0;
    //__threadfence();
    int i = blockIdx.x;
    if (hist1[i] + hist2[i] != 0){
        double temp = (double)((double)SQR(hist1[i] - hist2[i])) / (hist1[i] + hist2[i]);
        atomicAdd((float*)distance,(float)temp);
    }
}

__global__ void image_to_hisogram_shared(uchar *image1, OUT int *hist1) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    __shared__ uchar im[IMAGE_SIZE];
    __shared__ int sharedHist[256];
    if (i*32+j <256){
        sharedHist[i*32+j] = 0;
    };
    im[i*32+j]=image1[i*32+j];
    threadfence();
    uchar pattern = local_binary_pattern(im, i, j);
    atomicAdd(sharedHist+pattern,1);
    threadfence();
    if (i*32+j <256){
        hist1[i*32+j] = sharedHist[i*32+j];
    };
}


int main() {
    uchar *images1; /* we concatenate all images in one huge array */
    uchar *images2;
    CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );

    load_image_pairs(images1, images2);
    double t_start, t_finish;
    double total_distance;

    /* using CPU */
    printf("\n=== CPU ===\n");
    int histogram1[256];
    int histogram2[256];
    t_start  = get_time_msec();
    for (int i = 0; i < N_IMG_PAIRS; i++) {
        image_to_histogram(&images1[i * IMG_DIMENSION * IMG_DIMENSION], histogram1);
        image_to_histogram(&images2[i * IMG_DIMENSION * IMG_DIMENSION], histogram2);
        total_distance += histogram_distance(histogram1, histogram2);
    }
    t_finish = get_time_msec();
    printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    printf("total time %f [msec]\n", t_finish - t_start);

    /* using GPU task-serial */
    printf("\n=== GPU Task Serial ===\n");
    do {
        //* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise *//*
        //* Your Code Here *//*
        uchar *gpu_image1, *gpu_image2; // TODO: allocate with cudaMalloc
        cudaMalloc(&gpu_image1,1024*sizeof(uchar));
        cudaMalloc(&gpu_image2,1024*sizeof(uchar));
        int *gpu_hist1, *gpu_hist2; // TODO: allocate with cudaMalloc
        cudaMalloc(&gpu_hist1,256*sizeof(int));
        cudaMalloc(&gpu_hist2,256*sizeof(int));
        cudaMemset(&gpu_hist1,0,256*sizeof(int));
        cudaMemset(&gpu_hist2,0,256*sizeof(int));
        double *gpu_hist_distance; //TODO: allocate with cudaMalloc
        cudaMalloc(&gpu_hist_distance,sizeof(double));
        double cpu_hist_distance;

        t_start = get_time_msec();
        for (int i = 0; i < N_IMG_PAIRS; i++) {
            dim3 threadsPerBlock(32,32);
            // TODO: copy relevant images from images1 and images2 to gpu_image1 and gpu_image2
            cudaMemcpy(gpu_image1, images1, 1024 * sizeof(uchar), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_image2, images2, 1024 * sizeof(uchar), cudaMemcpyHostToDevice);

            image_to_hisogram_simple<<<1, threadsPerBlock>>>(gpu_image2, gpu_hist2);
            image_to_hisogram_simple<<<1, threadsPerBlock>>>(gpu_image1, gpu_hist1);
            histogram_distance<<<1, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);
            //TODO: copy gpu_hist_distance to cpu_hist_distance 
            cudaMemcpy(&cpu_hist_distance, gpu_hist_distance, sizeof(double), cudaMemcpyDeviceToHost);
            
            total_distance += cpu_hist_distance;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t_finish = get_time_msec();
        printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
        printf("total time %f [msec]\n", t_finish - t_start);
    } while (0);

    /* using GPU task-serial + images and histograms in shared memory */
    printf("\n=== GPU Task Serial with shared memory ===\n");
    do { /* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise */
        /* Your Code Here */
        uchar *gpu_image1;
        uchar *gpu_image2; // TODO: allocate with cudaMalloc
        cudaMalloc(&gpu_image1,1024*sizeof(uchar));
        cudaMalloc(&gpu_image2,1024*sizeof(uchar));
        int *gpu_hist1;
        int *gpu_hist2; // TODO: allocate with cudaMalloc
        cudaMalloc(&gpu_hist1,256*sizeof(int));
        cudaMalloc(&gpu_hist2,256*sizeof(int));
        //cudaMemset(&gpu_hist1,0,256*sizeof(int));
        //cudaMemset(&gpu_hist2,0,256*sizeof(int));
        double *gpu_hist_distance; //TODO: allocate with cudaMalloc
        cudaMalloc(&gpu_hist_distance,sizeof(double));
        double cpu_hist_distance;

        t_start = get_time_msec();
        for (int i = 0; i < N_IMG_PAIRS; i++) {
            dim3 threadsPerBlock(32,32);
            // TODO: copy relevant images from images1 and images2 to gpu_image1 and gpu_image2
            cudaMemcpy(gpu_image1, images1, 1024 * sizeof(uchar), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_image2, images2, 1024 * sizeof(uchar), cudaMemcpyHostToDevice);

            image_to_hisogram_shared<<<1, threadsPerBlock>>>(gpu_image1, gpu_hist1);
            image_to_hisogram_shared<<<1, threadsPerBlock>>>(gpu_image2, gpu_hist2);
            //->move to global hiat

            histogram_distance<<<1, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);
            //TODO: copy gpu_hist_distance to cpu_hist_distance
            cudaMemcpy(&cpu_hist_distance, gpu_hist_distance, sizeof(double), cudaMemcpyDeviceToHost);

            total_distance += cpu_hist_distance;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t_finish = get_time_msec();
        printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
        printf("total time %f [msec]\n", t_finish - t_start);
    } while (0);
    printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    printf("total time %f [msec]\n", t_finish - t_start);

    /* using GPU + batching */
    printf("\n=== GPU Batching ===\n");
    /* Your Code Here */
    printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    printf("total time %f [msec]\n", t_finish - t_start);

    return 0;
}
//bla

