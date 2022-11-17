#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <cstring>
#include <windows.h>
#include <math.h>

#include "save.h"

#define DEBUGTARGET 0

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
    if (cnt > 0) {
        printf("%d elements do not match. ", cnt);
        printf("Arrays do not match.\n\n");
    } else {
        printf("Arrays match!\n");
    }
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

__global__ void createConeGPU(float *g_cone, int Nx, int Ny) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = ix * Ny + iy;

    if (ix < Nx && iy < Ny) {
        g_cone[idx] = 0;
        for (int k = 0; k < 400; k++) {
            if ((ix - 512) * (ix - 512) + (iy - 700) * (iy - 700) < k * k) {
                g_cone[idx]++;
            }
        }
    }
}

__global__ void shadowComputeGPU(float *g_shadow, int Nx, int Ny, float *M, float m, float Y_s0, float Z_s0) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = ix * blockDim.y + iy;

    if (x < Nx) {
        // 记录本行的最大下视角
        float maxTheta = 0;
        g_shadow[x * Ny + 0] = 1;

        for (int y = 0; y < Ny - 1; y++) {
            float y0 = y * m;
            float z0 = M[x * Ny + y];
            // 下视角 arctan((y0 - Y_s0)/(Z_s0 - z0))
            float theta = (y0 - Y_s0) / (Z_s0 - z0);

            if (maxTheta < theta) {
                g_shadow[x * Ny + (y + 1)] = 1;
                maxTheta = theta;
            } else {
                g_shadow[x * Ny + (y + 1)] = 0;
            }
        }
    }
}

__global__ void backscatterComputeGPU(float *g_sigma, int Nx, int Ny, float *g_M, float m, float Z_s0) {
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
        // g_sigma[x * Ny + y] = 10 * log10(0.0133 * cos(theta) / pow((sin(theta) + 0.1 * cos(theta)), 3));

        // currie: 《典型地形的 SAR 回波模拟及快速实现》张吉宇 p25
        float S = m * m * sqrt(a * a + b * b + 1);
        float avgh = (z1 + z2 + z3 + z4) / 4;
        float sigma_h = sqrt((z1 - avgh) * (z1 - avgh) + (z2 - avgh)  * (z2 - avgh) + (z3 - avgh) * (z3 - avgh) + (z4 -avgh) * (z4 - avgh)) / 3;
        // 树林
        float A = 0.00054;
        float B = 0.64;
        float C = 0.02;
        float D = 0;

        float lambda = 0.06;

        float sigma_a = A * pow(C + theta, B) * exp(-D * lambda / (0.1 * sigma_h + lambda));
        g_sigma[x * Ny + y] = sigma_a * S * cos(theta) * cos(theta);
    }
}

__global__ void imageSimulationGPU(float *g_img, int Nx, int Ny, float *g_M, float *g_sigma, float *g_shadow,
                                   float m, float Y_s0, float Z_s0, float R_0, float M_s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx - 1 && y < Ny - 1) {
        // Leberl 构像模型计算模拟 SAR 图像的纵坐标
        int Y = ceil(
                (sqrt((y * m - Y_s0) * (y * m - Y_s0) + (g_M[x * Ny + y] - Z_s0) * (g_M[x * Ny + y] - Z_s0)) - R_0) /
                M_s);
        if (Y > 0 && Y < Ny) {
            atomicAdd(&g_img[x * Ny + Y], g_sigma[x * Ny + y] * g_shadow[x * Ny + y]);
//            g_img[x * Ny + Y] += g_sigma[x * Ny + y] * g_shadow[x * Ny + y];
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
    float *img = (float *) malloc(nBytes);
    memset(img, 0, nBytes);

    // allocate device memory
    float *g_img = NULL;
    CHECK(cudaMalloc((void **) &g_img, nBytes));

    // execution configuration

    // kernel: imageSimulationGPU
    QueryPerformanceCounter(&iStart);

    imageSimulationGPU<<<grid, block>>>(g_img, nx, ny, g_M, g_sigma, g_shadow, m, Y_s0, Z_s0, R_0, M_s);

    CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&iEnd);
    printf("\t[ imageSimulationGPU\t<<<(%d,%d), (%d,%d)>>> ] \telapsed %f s\n",
           grid.x, grid.y, block.x, block.y,
           (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart
    );

    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(img, g_img, nBytes, cudaMemcpyDeviceToHost));

    /*-----------------------------------------------------------------------*/
    // 归一化为灰度图像

    // 保存
    saveMatrix(img, nx, ny, "img");

    free(cone);
    free(shadow);
    free(sigma);
    free(img);

    CHECK(cudaFree(g_cone));
    CHECK(cudaFree(g_shadow));
    CHECK(cudaFree(g_sigma));
    CHECK(cudaFree(g_img));
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
//            sigma[x * ny + y] = 10 * log10(0.0133 * cos(theta) / pow((sin(theta) + 0.1 * cos(theta)), 3));

            // currie: 《典型地形的 SAR 回波模拟及快速实现》张吉宇 p25
            float S = m * m * sqrt(a * a + b * b + 1);
            float avgh = (z1 + z2 + z3 + z4) / 4;
            float sigma_h = sqrt((z1 - avgh) * (z1 - avgh) + (z2 - avgh)  * (z2 - avgh) + (z3 - avgh) * (z3 - avgh) + (z4 -avgh) * (z4 - avgh)) / 3;
            // 树林
            float A = 0.00054;
            float B = 0.64;
            float C = 0.02;
            float D = 0;

            float lambda = 0.06;

            float sigma_a = A * pow(C + theta, B) * exp(-D * lambda / (0.1 * sigma_h + lambda));
            sigma[x * ny + y] = sigma_a * S * cos(theta) * cos(theta);
        }
    }

    QueryPerformanceCounter(&iEnd);
    printf("Backscattering coefficient computing elapsed %f s\n",
           (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart);

    /*------------------------------SAR模拟-----------------------------------*/
    QueryPerformanceCounter(&iStart);

    float *img = (float *) malloc(nBytes);
    memset(img, 0, nBytes);

    for (int x = 0; x < nx - 1; x++) {
        for (int y = 0; y < ny - 1; y++) {
            // Leberl 构像模型计算模拟 SAR 图像的纵坐标
            int Y = ceil(
                    (sqrt((y * m - Y_s0) * (y * m - Y_s0) + (M[x * ny + y] - Z_s0) * (M[x * ny + y] - Z_s0)) - R_0) /
                    M_s);
            if (Y > 0 && Y < ny) {
                img[x * ny + Y] += sigma[x * ny + y] * shadow[x * ny + y];
            }
        }
    }

    QueryPerformanceCounter(&iEnd);
    printf("SAR image simulation elapsed %f s\n", (double) (iEnd.QuadPart - iStart.QuadPart) / (double) tc.QuadPart);

    /*---------------------------------处理+保存--------------------------------------*/
    // 归一化为灰度图像...

    // 保存
    saveMatrix(img, nx, ny, "img");


//    /*--------------------------------回波模拟----------------------------------------*/
//    printf("echo simulating...\n");
//    /*--------------------------------数据预处理--------------------------------------*/
//
//    float *mat = (float *) malloc(nBytes);
//    for (int i = 0; i < nx; i++) {
//        for (int j = 0; j < ny; j++) {
//            mat[i * ny + j] = sigma[i * ny + j] * shadow[i * ny + j];
//        }
//    }
//    saveMatrix(mat, nx, ny, "mat");
//
//#ifdef DEBUGTARGET
//    nxy = 3;
//    nBytes = nxy * sizeof(float);
//    float *target = (float *) malloc(nBytes * 4);
//    memset(target, 0, sizeof target);
//
//    target[0] = 0;
//    target[1] = 0;
//    target[2] = 0;
//    target[3] = 2;
//    target[4] = 80;
//    target[5] = 45;
//    target[6] = 0;
//    target[7] = -2;
//    target[8] = -20;
//    target[9] = -20;
//    target[10] = 0;
//    target[11] = 10;
//
//#else
//    float *target = (float *) malloc(nBytes * 4);
//    memset(target, 0, sizeof target);
//    int idx = 0;
//    for (int i = 0; i < nx; i++) {
//        for (int j = 0; j < ny; j++) {
//            target[idx * 4 + 0] = j - ny / 2;
//            target[idx * 4 + 1] = i - nx / 2;
//            target[idx * 4 + 2] = M[i * ny + j];
//            target[idx * 4 + 3] = sigma[i * ny + j] * shadow[i * ny + j];
//        }
//    }
//#endif
//
//
//    /*--------------------------------雷达参数----------------------------------------*/
//    float c = 3e8; // 光速
//    float fc = 5e9; // 载波频率
//    float B = 200e6; // 带宽
//
//    float lambda = c / fc;   // 波长
//    float Tp = 1.5e-6;    // 脉宽
//    float Kr = B / Tp;    // 调频率
//    float fs = 1.6 * B;     // 采样率
//
//    float H = 1730;     // 飞机高度
//    float Ls = 200;     // 合成孔径长度
//    float v = 100;      // 飞机速度
//    float Lt = Ls / v;    // 合成孔径时间
//
//    // 成像区域[Xc-X0,Xc+X0; Yc-Y0,Yc+Y0]
//    // 以合成孔径中心为原点，距离向为x轴，方位向为y轴
//    float Xc = 2000;
//    float Yc = 0;
//    float Xo = 200;
//    float Yo = 200;
//
//
//    for (int i = 0; i < nxy; i++) {
//        target[i * 4 + 0] *= Xc;
//        target[i * 4 + 1] *= Yc;
//    }
//
//    float Rc = sqrt(H * H + Xc * Xc);     // 中心距离
//    float Ka = 2 * v * v / (Rc * lambda);   // 多普勒调频率
//    float Bmax = Lt * Ka;            // 多普勒最大带宽
//    float fa = ceil(3 * Bmax);       // 脉冲重复频率
//
//    float Rmin = sqrt(H * H + (Xc - Xo) * (Xc - Xo));   // 观测场景距飞机的最近距离
//    float Rmax = sqrt((Xc + Xo) * (Xc + Xo) + H * H + (Yc + Yo + Ls / 2) * (Yc + Yo + Ls / 2));  // 最远距离
//    float rm = Ls + 2 * Yo;    // 雷达走过的总路程长度
//
//    int len_tm = (rm * fa) / v;
//    float *tm = (float *) malloc(len_tm * sizeof(float));                  // 慢时间（合成孔径时间+成像区域时间）
//    for (int i = 0; i < len_tm; i++) {
//        tm[i] = i / fa;
//    }
//
//    int len_tk = (2 * Rmax / c - 2 * Rmin / c + Tp) * fs;
//    float *tk = (float *) malloc(len_tk * sizeof(float));       // 快时间（距离门内）
//    for (int i = 0; i < len_tk; i++) {
//        tk[i] = (2 * Rmin / c - Tp / 2) + i / fs;
//    }
//
//    float *echo_all = (float *) malloc(len_tm * len_tk * sizeof(float)); // 回波
//    float *y = (float *) malloc(len_tm * sizeof(float)); // 飞机y轴坐标
//    for (int i = 0; i < len_tm; i++) {
//        y[i] = -v * (rm / v) / 2 + v * tm[i];
//    }
//
//    /*
//    for (int k = 0; k < nxy; k++) { // 目标数
//        for (int i = 0; i < len_tm; i++) { // 慢时间轴
//            sigma = target(k, 4);
//            if (sigma == 0) continue;
////
////            radar = [0,y(i),H ];      %飞机坐标
////            Rtm = sqrt(sum((target(k,1:3)-radar).^2));
////            echo_all(i,:) = echo_all(i,:) + sigma * (abs(target(k,2)-y(i))/Xc < 0.01)*rectpuls(tk-2*Rtm/c,Tp).* exp(1j*2*pi*fc*(tk-2*Rtm/c)+1j*pi*Kr*(tk-2*Rtm/c).^2);  %回波模型
//        }
//    }
//     */

    free(cone);
    free(shadow);
    free(sigma);
    free(img);
//    free(target);
//    free(tm);
//    free(tk);
//    free(echo_all);
}
