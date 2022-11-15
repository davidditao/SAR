#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <cstring>
#include <windows.h>

#include "save.h"

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

void checkResultGPU(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;

    int printLimit = 1;
    int cnt = 0;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            if (printLimit > 0) {
                printf("host(%d) %f, gpu(%d) %f. ", i, hostRef[i], i, gpuRef[i]);
                printLimit--;
            }
            cnt++;
        }
    }
    printf("%d elements do not match. ", cnt);
    printf("Arrays do not match.\n\n");
}

void checkResultCPU(float **host2d, float *host1d, const int nx, const int ny) {
    double epsilon = 1.0E-8;

    int idx = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            if (abs(host2d[i][j] - host1d[idx]) > epsilon) {
                printf("host2d(%d, %d) %f, host1d(%d) %f ", i, j, host2d[i][j], idx, host1d[idx]);
                printf("Arrays do not match.\n\n");
                break;
            }
            idx++;
        }
    }
}

float **malloc2d(int row, int col) {
    float **ret = (float **) malloc(sizeof(float *) * row);  // 指向二维数组，一维数组指针的二级指针
    float *p = (float *) malloc(sizeof(float) * row * col);   // 为二维数组的每个元素分配空间
    memset(p, 0, sizeof(float) * row * col);

    // 分别为一维数组指针指向所拥有的二维数组元素的控件
    int i = 0;
    if (ret && p) //安全检查
    {
        for (i = 0; i < row; i++) {
            ret[i] = (p + i * col);  // ret[i] 是二维数组的一维数组指针  ret[i]  = *(ret + i)
        }
    } else {
        free(ret);
        free(p);
        ret = NULL;
        p = NULL;
    }

    return ret;
}

void free2d(float **a) {
    free(a[0]); //对应 p, 必须先释放他！！
    free(a);   //对应 ret
}

void showMatrix2(float **matrix, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

__global__ void createConeGPU(float *d_cone, int Nx, int Ny) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = ix * Ny + iy;

    if (ix < Nx && iy < Ny) {
        d_cone[idx] = 0;
        for (int k = 0; k < 400; k++) {
            if ((ix - 512) * (ix - 512) + (iy - 512) * (iy - 512) < k * k) {
                d_cone[idx]++;
            }
        }
    }
}

__global__ void shadowComputeGPU(float *d_shadow, int Nx, int Ny, float *M, float m, float Y_s0, float Z_s0) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = ix * blockDim.y + iy;

    if (x < Nx) {
        // 记录本行的最大下视角
        float maxTheta = 0;
        d_shadow[x * Ny + 0] = 1;

        for (int y = 0; y < Ny - 1; y++) {
            float y0 = y * m;
            float z0 = M[x * Ny + y];
            // 下视角 arctan((y0 - Y_s0)/(Z_s0 - z0))
            float theta = (y0 - Y_s0) / (Z_s0 - z0);

            if (maxTheta < theta) {
                d_shadow[x * Ny + (y + 1)] = 1;
                maxTheta = theta;
            } else {
                d_shadow[x * Ny + (y + 1)] = 0;
            }
        }
    }
}

__global__ void backscatterComputeGPU(float *d_sigma, int Nx, int Ny, float *g_M, float m, float Z_s0) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//    unsigned int idx = ix * Ny + iy;

    if (x < Nx - 1 && y < Ny - 1) {
        // 小面元，取四个点：(0, 0, z1) (m, 0, z2) (m, m, z3) (0, m, z4)，
        // 最小二乘法拟合平面：ax + by + c
        // 论文：《利用 RD 模型和 DEM 数据的高分辨率机载 SAR 图像模拟》 王庆
        float z1 = g_M[x * Ny + y];
        float z2 = g_M[(x + 1) * Ny + y];
        float z3 = g_M[(x + 1) * Ny + (y + 1)];
        float z4 = g_M[x * Ny + (y + 1)];
        // 解得：
        float a = (z2 + z3 - z1 - z4) / (2 * m);
        float b = (z3 + z4 - z1 - z2) / (2 * m);
        float c = (3 * z1 + z2 + z4 - z3) / 4;

        // 拟合平面的法向量：n = (a, b, -1)
        // n 与 z 轴的夹角，即坡度角 gama
        float gama = acos(1 / sqrt(a * a + b * b + 1));

        // 小面元中心点高程
        float z0 = a * (m / 2) + b * (m / 2) + c;

        int d = y; // 该点的地距

        // 雷达对小面元的入射角:
        float theta1 = atan(d * m / (Z_s0 - z0));

        // 局部入射角 theta : [-pi/2, pi/2]
        float theta = acos((abs(b * sin(theta1) + cos(theta1))) / sqrt(a * a + b * b + 1));

        // 后向散射系数：Muhleman 半经验模型
        d_sigma[x * Ny + y] = 10 * log10(0.0133 * cos(theta) / pow((sin(theta) + 0.1 * cos(theta)), 3));
    }
}

__global__ void imageSimulationGPU(float *g_signal, int Nx, int Ny, float *g_M, float *g_sigma, float *g_shadow,
                                   float m, float Y_s0, float Z_s0, float R_0, float M_s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx - 1 && y < Ny - 1) {
        // Leberl 构像模型计算模拟 SAR 图像的纵坐标
        int Y = ceil(
                (sqrt((y * m - Y_s0) * (y * m - Y_s0) + (g_M[x * Ny + y] - Z_s0) * (g_M[x * Ny + y] - Z_s0)) - R_0) /
                M_s);
        if (Y > 0 && Y < Ny) {
            atomicAdd(&g_signal[x * Ny + Y], g_sigma[x * Ny + y] * g_shadow[x * Ny + y]);
//            g_signal[x * Ny + Y] += g_sigma[x * Ny + y] * g_shadow[x * Ny + y];
        }
    }
}

void GPU() {
    /*-----------------------------set up device---------------------------*/
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using GPU Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set timer
    LARGE_INTEGER iStart, iEnd, tc;
    QueryPerformanceFrequency(&tc);

    /*---------------------------create a cone model----------------------*/
    const int cone_row = 1024, cone_col = 2000;
    int nx = cone_row, ny = cone_col;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    float *cone = (float *) malloc(nBytes);
    memset(cone, 0, nBytes);

    // allocate device memory
    float *g_cone = NULL;
    cudaMalloc((void **) &g_cone, nBytes);

    // execution configuration
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // kernel: createConeGPU
    QueryPerformanceCounter(&iStart);

    createConeGPU<<<grid, block>>>(g_cone, cone_row, cone_col);

    CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&iEnd);
    printf("\t[ createConeGPU\t\t<<<(%d,%d), (%d,%d)>>> ] \telapsed %f s\n",
           grid.x, grid.y, block.x, block.y,
           (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart
    );

    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(cone, g_cone, nBytes, cudaMemcpyDeviceToHost));

    /*------------------------------雷达参数-----------------------------------*/

    float *M = cone; // load data
    float *g_M = g_cone;

    // 雷达初始位置
    float X_s0 = 0;
    float Y_s0 = -2000;
    float Z_s0 = 1730;

    float R_0 = sqrt((Y_s0 / 2) * (Y_s0 / 2) + Z_s0 * Z_s0); // 近距延迟 sqrt((y/2)^2 + z^2)

    int M_s = 1; // 斜距向采样间隔
    int M_a = 1; // 方位向采样间隔

    float m = 1; // DEM 两点间隔

    /*---------------------------------阴影计算-------------------------------*/
    float *shadow = (float *) malloc(nBytes);
    memset(shadow, 0, nBytes);

    // allocate device memory
    float *g_shadow = NULL;
    CHECK(cudaMalloc((void **) &g_shadow, nBytes));

    // execution configuration
    grid.x = (nx + block.x * block.y - 1) / (block.x * block.y);
    grid.y = 1;

    // kernel shadowComputeGPU
    QueryPerformanceCounter(&iStart);

    shadowComputeGPU<<<grid, block>>>(g_shadow, nx, ny, g_M, m, Y_s0, Z_s0);

    CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&iEnd);
    printf("\t[ shadowComputeGPU\t<<<(%d,%d), (%d,%d)>>> ] \t\telapsed %f s\n",
           grid.x, grid.y, block.x, block.y,
           (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart
    );

    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(shadow, g_shadow, nBytes, cudaMemcpyDeviceToHost));

    /*------------------------------后向散射系数计算---------------------------*/
    float *sigma = (float *) malloc(nBytes);
    memset(sigma, 0, nBytes);

    // allocate device memory
    float *g_sigma = NULL;
    CHECK(cudaMalloc((void **) &g_sigma, nBytes));

    // execution configuration
    grid.x = (nx + block.x - 1) / block.x;
    grid.y = (ny + block.y - 1) / block.y;

    // kernel: backscatterComputeGPU
    QueryPerformanceCounter(&iStart);

    backscatterComputeGPU<<<grid, block>>>(g_sigma, nx, ny, g_M, m, Z_s0);

    CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&iEnd);
    printf("\t[ backscatterComputeGPU\t<<<(%d,%d), (%d,%d)>>> ] \telapsed %f s\n",
           grid.x, grid.y, block.x, block.y,
           (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart
    );

    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(sigma, g_sigma, nBytes, cudaMemcpyDeviceToHost));

    /*------------------------------SAR模拟-----------------------------------*/
    // CPU
    QueryPerformanceCounter(&iStart);

    float *cpu_signal= (float *) malloc(nBytes);
    memset(cpu_signal, 0, nBytes);

    for (int x = 0; x < nx - 1; x++) {
        for (int y = 0; y < ny - 1; y++) {
            // Leberl 构像模型计算模拟 SAR 图像的纵坐标
            int Y = ceil(
                    (sqrt((y * m - Y_s0) * (y * m - Y_s0) + (M[x * ny + y] - Z_s0) * (M[x * ny + y] - Z_s0)) - R_0) /
                    M_s);
            if (Y > 0 && Y < ny) {
                cpu_signal[x * ny + Y] += sigma[x * ny + y] * shadow[x * ny + y];
            }
        }
    }

    QueryPerformanceCounter(&iEnd);
    printf("SAR image simulation elapsed %f s\n", (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart);

    //------
    float *signal = (float *) malloc(nBytes);
    memset(signal, 0, nBytes);

    // allocate device memory
    float *g_signal = NULL;
    CHECK(cudaMalloc((void **) &g_signal, nBytes));

    // execution configuration

    // kernel: imageSimulationGPU
    QueryPerformanceCounter(&iStart);

    imageSimulationGPU<<<grid, block>>>(g_signal, nx, ny, g_M, g_sigma, g_shadow, m, Y_s0, Z_s0, R_0, M_s);

    CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&iEnd);
    printf("\t[ imageSimulationGPU\t<<<(%d,%d), (%d,%d)>>> ] \telapsed %f s\n",
           grid.x, grid.y, block.x, block.y,
           (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart
    );

    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(signal, g_signal, nBytes, cudaMemcpyDeviceToHost));

    // check result
    checkResultGPU(cpu_signal, signal, nxy);

    /*-----------------------------------------------------------------------*/
    // 归一化为灰度图像

    // 保存
    saveMatrix(signal, nx, ny, "signal");

    free(cone);
    free(shadow);
    free(sigma);
    free(signal);

    CHECK(cudaFree(g_cone));
    CHECK(cudaFree(g_shadow));
    CHECK(cudaFree(g_sigma));
    CHECK(cudaFree(g_signal));
}

void CPU();

int main(int argc, char **argv) {
    if (argc > 1) {
        printf("|%s|\n", argv[1]);
        if (strcmp(argv[1], "CPU") == 0) {
            CPU();
        } else if (strcmp(argv[1], "GPU") == 0) {
            GPU();
        }
        return 0;
    }

    // default
//    CPU();
    GPU();
    return 0;
}

void CPU() {
    printf("using CPU\n");

    // set timer
    LARGE_INTEGER iStart, iEnd, tc;
    QueryPerformanceFrequency(&tc);

    /*---------------------------create a cone model----------------------*/
    QueryPerformanceCounter(&iStart);
    const int cone_row = 1024, cone_col = 2000;
    int nx = cone_row, ny = cone_col;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    float *cone = (float *) malloc(nBytes);

    memset(cone, 0, nBytes);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < 400; k++) {
                if ((i - 512) * (i - 512) + (j - 700) * (j - 700) < k * k) {
                    cone[i * ny + j]++;
                }
            }
        }
    }
    QueryPerformanceCounter(&iEnd);
    printf("create cone elapsed %f s\n", (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart);

//    saveMatrix(cone, nx, ny, "gpuCone");

    /*------------------------------雷达参数-----------------------------------*/

    float *M = cone; // load data

    // 雷达初始位置
    float X_s0 = 0;
    float Y_s0 = -2000;
    float Z_s0 = 1730;

    float R_0 = sqrt((Y_s0 / 2) * (Y_s0 / 2) + Z_s0 * Z_s0); // 近距延迟 sqrt((y/2)^2 + z^2)

    int M_s = 1; // 斜距向采样间隔
    int M_a = 1; // 方位向采样间隔

    float m = 1; // DEM 两点间隔

    /*---------------------------------阴影计算-------------------------------*/
    QueryPerformanceCounter(&iStart);

    float *shadow = (float *) malloc(nBytes);
    memset(shadow, 0, nBytes);

    for (int x = 0; x < nx; x++) {
        // 记录本行的最大下视角
        float maxTheta = 0;
        shadow[x * ny + 0] = 1;

        for (int y = 0; y < ny - 1; y++) {
            float y0 = y * m;
            float z0 = M[x * ny + y];
            // 下视角 arctan((y0 - Y_s0)/(Z_s0 - z0))
            float theta = (y0 - Y_s0) / (Z_s0 - z0);

            if (maxTheta < theta) {
                shadow[x * ny + (y + 1)] = 1;
                maxTheta = theta;
            } else {
                shadow[x * ny + (y + 1)] = 0;
            }
        }
    }

    QueryPerformanceCounter(&iEnd);
    printf("shadow computing elapsed %f s\n", (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart);

    /*------------------------------后向散射系数计算---------------------------*/
    QueryPerformanceCounter(&iStart);

    float *sigma = (float *) malloc(nBytes);
    memset(sigma, 0, nBytes);

    for (int x = 0; x < nx - 1; x++) {
        for (int y = 0; y < ny - 1; y++) {
            // 小面元，取四个点：(0, 0, z1) (m, 0, z2) (m, m, z3) (0, m, z4)，
            // 最小二乘法拟合平面：ax + by + c
            // 论文：《利用 RD 模型和 DEM 数据的高分辨率机载 SAR 图像模拟》 王庆
            float z1 = cone[x * ny + y];
            float z2 = cone[(x + 1) * ny + y];
            float z3 = cone[(x + 1) * ny + (y + 1)];
            float z4 = cone[x * ny + (y + 1)];
            // 解得：
            float a = (z2 + z3 - z1 - z4) / (2 * m);
            float b = (z3 + z4 - z1 - z2) / (2 * m);
            float c = (3 * z1 + z2 + z4 - z3) / 4;

            // 拟合平面的法向量：n = (a, b, -1)
            // n 与 z 轴的夹角，即坡度角 gama
            float gama = acos(1 / sqrt(a * a + b * b + 1));

            // 小面元中心点高程
            float z0 = a * (m / 2) + b * (m / 2) + c;

            int d = y; // 该点的地距

            // 雷达对小面元的入射角:
            float theta1 = atan(d * m / (Z_s0 - z0));

            // 局部入射角 theta : [-pi/2, pi/2]
            float theta = acos((abs(b * sin(theta1) + cos(theta1))) / sqrt(a * a + b * b + 1));

            // 后向散射系数：Muhleman 半经验模型
            sigma[x * ny + y] = 10 * log10(0.0133 * cos(theta) / pow((sin(theta) + 0.1 * cos(theta)), 3));
        }
    }

    QueryPerformanceCounter(&iEnd);
    printf("Backscattering coefficient computing elapsed %f s\n",
           (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart);

    /*------------------------------SAR模拟-----------------------------------*/
    QueryPerformanceCounter(&iStart);

    float *signal = (float *) malloc(nBytes);
    memset(signal, 0, nBytes);

    for (int x = 0; x < nx - 1; x++) {
        for (int y = 0; y < ny - 1; y++) {
            // Leberl 构像模型计算模拟 SAR 图像的纵坐标
            int Y = ceil(
                    (sqrt((y * m - Y_s0) * (y * m - Y_s0) + (M[x * ny + y] - Z_s0) * (M[x * ny + y] - Z_s0)) - R_0) /
                    M_s);
            if (Y > 0 && Y < ny) {
                signal[x * ny + Y] += sigma[x * ny + y] * shadow[x * ny + y];
            }
        }
    }

    QueryPerformanceCounter(&iEnd);
    printf("SAR image simulation elapsed %f s\n", (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart);

    /*-----------------------------------------------------------------------*/
    // 归一化为灰度图像

    // 保存
    saveMatrix(signal, nx, ny, "signal");

    free(cone);
    free(shadow);
    free(sigma);
    free(signal);
}
