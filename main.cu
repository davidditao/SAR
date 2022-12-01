#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <cstring>
#include <windows.h>
#include <math.h>

#include "save.h"

// 开启回波模拟
#define ECHO_SIMU_GPU

const float C = 3e8;                        // 光速
const float Fc = 5e9;                       // 载波频率
const float lambda = C / Fc;                // 波长
const float B = 200e6;                      // 带宽
const float Tp = 1.5e-6;                    // 脉宽
const float PI = 3.14159265358979323846;    // pi

__device__ int target_count;

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

// 均值滤波
float* imfliter(float *A, int nx, int ny) {
    float *B = (float *) malloc(nx * ny * sizeof(float));

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            float a00 = 0, a01 = 0, a02 = 0, a10 = 0, a11 = 0, a12 = 0, a20 = 0, a21 = 0, a22 = 0;
            if (i - 1 >= 0) {
                if (j - 1 >= 0) {
                    a00 = A[(i - 1) * ny + (j - 1)];
                }

                a01 = A[(i - 1) * ny + j];

                if (j + 1 < ny) {
                    a02 = A[(i - 1) * ny + (j + 1)];
                }
            }

            if (j - 1 >= 0) {
                a10 = A[i * ny + (j - 1)];
            }

            a11 = A[i * ny + j];

            if (j + 1 < ny) {
                a12 = A[i * ny + (j + 1)];
            }

            if (i + 1 < nx) {
                if (j - 1 >= 0) {
                    a20 = A[(i + 1) * ny + (j - 1)];
                }

                a21 = A[(i + 1) * ny + j];

                if (j + 1 < ny) {
                    a22 = A[(i + 1) * ny + (j + 1)];
                }
            }

            B[i * ny + j] = (a00 + a01 + a02 + a10 + a11 + a12 + a20 + a21 + a22) / 9;
        }
    }

    return B;
}

// 归一化
void normalized(float *img, int nx, int ny) {
    float minVal = 1e5, maxVal = -1e5;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            if (img[i * ny + j] < minVal) {
                minVal = img[i * ny + j];
            }
            if (img[i * ny + j] > maxVal) {
                maxVal = img[i * ny + j];
            }

        }
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            img[i * ny + j] = (img[i * ny + j] - minVal) * 255 / (maxVal - minVal);
        }
    }
}

void checkResultGPU(float *hostRef, float *gpuRef, const int N) {
    float epsilon = 1.0E-8;

    int printLimit = 1;
    int cnt = 0;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            if (printLimit > 0) {
                printf("host(%d) %.10f, gpu(%d) %.10f. ", i, hostRef[i], i, gpuRef[i]);
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
    float epsilon = 1.0E-8;

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
            if ((ix - 512) * (ix - 512) + (iy - 512) * (iy - 512) < k * k) {
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

__global__ void backscatterComputeGPU(float *g_sigma, int Nx, int Ny, float *g_M, float m, float Z_s0, float lambda) {
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
        float theta;
        float dividend = abs(b * sin(theta1) + cos(theta1));
        float divisor = sqrt(a * a + b * b + 1);
        if (abs(dividend - divisor) < 1e-6) { // 防止出现 -nan
            theta = 0;
        } else {
            theta = acos(dividend / divisor);
        }

        // 后向散射系数：Muhleman 半经验模型
        // g_sigma[x * Ny + y] = 10 * log10(0.0133 * cos(theta) / pow((sin(theta) + 0.1 * cos(theta)), 3));

        // currie: 《典型地形的 SAR 回波模拟及快速实现》张吉宇 p25
        float S = m * m * sqrt(a * a + b * b + 1); // 小面元的截面积
        float avgh = (z1 + z2 + z3 + z4) / 4; // 高度平均值
        float sigma_h = sqrt((z1 - avgh) * (z1 - avgh) + (z2 - avgh) * (z2 - avgh) + (z3 - avgh) * (z3 - avgh) +
                             (z4 - avgh) * (z4 - avgh)) / 3; // 标准差

        // 树林
        float A = 0.00054;
        float B = 0.64;
        float C = 0.02;
        float D = 0;

//        float lambda = C / Fc;   // 波长

        float sigma_a = A * pow(C + theta, B) * exp(-D * lambda / (0.1 * sigma_h + lambda));
        g_sigma[x * Ny + y] = sigma_a * S * cos(theta) * cos(theta);
    }
}

__global__ void imageSimulationGPU(float *g_img, int imgNy, int miny, int delta, int Nx, int Ny,
                                   float *g_M, float *g_sigma, float *g_shadow,
                                   float m, float Y_s0, float Z_s0, float R_0, float M_s) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx - 1 && y < Ny - 1) {
        // Leberl 构像模型计算模拟 SAR 图像的纵坐标
        int Y = ceil(
                (sqrt((y * m - Y_s0) * (y * m - Y_s0) + (g_M[x * Ny + y] - Z_s0) * (g_M[x * Ny + y] - Z_s0)) - R_0) /
                M_s);
        atomicAdd(&g_img[x * imgNy + (Y - miny + delta)], g_sigma[x * Ny + y] * g_shadow[x * Ny + y]);
    }
}

__global__ void createEchoGPU(float *g_real, float *g_imag, float *g_target, float *g_tk, float *g_y,
                              int target_size, int tk_size, float H, float Kr, float Xc,
                              float C, float Fc, float PI, float Tp) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int k = 0; k < target_size; k++) { // 目标数
        float Xs = g_target[k * 4 + 0];
        float Ys = g_target[k * 4 + 1];
        float Zs = g_target[k * 4 + 2];
        float sigma = g_target[k * 4 + 3];

        if (sigma == 0) continue;

        float Rtm = sqrt(Xs * Xs + (Ys - g_y[i]) * (Ys - g_y[i]) + (Zs - H) * (Zs - H));

        if ((abs(g_target[k * 4 + 1] - g_y[i]) / Xc) < 0.01) { // 假定波束宽度是0.02°
            float rec = 0;
            if (abs(g_tk[j] - 2 * Rtm / C) < Tp / 2) {
                rec = 1;
            }
            float exp_real = cos(2 * PI * Fc * (g_tk[j] - 2 * Rtm / C) +
                                 PI * Kr * (g_tk[j] - 2 * Rtm / C) * (g_tk[j] - 2 * Rtm / C));
            float exp_imag = sin(2 * PI * Fc * (g_tk[j] - 2 * Rtm / C) +
                                 PI * Kr * (g_tk[j] - 2 * Rtm / C) * (g_tk[j] - 2 * Rtm / C));
            atomicAdd(&g_real[i * tk_size + j], sigma * rec * exp_real);
            atomicAdd(&g_imag[i * tk_size + j], sigma * rec * exp_imag);
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
    const int cone_row = 1024, cone_col = 1400;
    int nx = cone_row, ny = cone_col;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    float *M = (float *) malloc(nBytes);
    memset(M, 0, nBytes);

    // allocate device memory
    float *g_M = NULL;
    cudaMalloc((void **) &g_M, nBytes);

    // execution configuration
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // kernel: createConeGPU
    QueryPerformanceCounter(&iStart);

    createConeGPU<<<grid, block>>>(g_M, nx, ny);

    CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&iEnd);
    printf("\t[ createConeGPU\t\t<<<(%d,%d), (%d,%d)>>> ] \telapsed %f s\n",
           grid.x, grid.y, block.x, block.y,
           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart
    );

    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(M, g_M, nBytes, cudaMemcpyDeviceToHost));

//    saveMatrix(M, nx, ny, "tmp");

//    /*------------------------------导入数据-----------------------------------*/
//    QueryPerformanceCounter(&iStart);
//
//    int nx = 0, ny = 0;
//    vector<float> DEM = load(nx, ny);
//    int nxy = nx * ny;
//    int nBytes = nxy * sizeof(float);
//
//    float *M = (float *) malloc(nBytes);
//
//    for (int i = 0; i < nxy; i++) {
//        M[i] = DEM[i];
//    }
//
//    // copy data to device side
//    float *g_M = NULL;
//    CHECK(cudaMalloc((void **) &g_M, nBytes));
//    CHECK(cudaMemcpy(g_M, M, nBytes, cudaMemcpyHostToDevice));
//
//    QueryPerformanceCounter(&iEnd);
//    printf("\t[ loading DEM data ] \t\t\t\t\telapsed %f s\n\n",
//           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart);
//
//    /*------------------------------GPU参数-----------------------------------*/
//    // execution configuration
//    int dimx = 32;
//    int dimy = 32;
//    dim3 block(dimx, dimy);
//    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    /*------------------------------雷达参数-----------------------------------*/
    // 雷达初始位置
    float X_s0 = 0;
    float Y_s0 = -2000;
    float Z_s0 = 1730;

//    float Y_s0 = -5000;
//    float Z_s0 = 5000;

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
    printf("\t[ shadowComputeGPU\t<<<(%d,%d), (%d,%d)>>> ] \t\telapsed %f s\n\n",
           grid.x, grid.y, block.x, block.y,
           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart
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

    backscatterComputeGPU<<<grid, block>>>(g_sigma, nx, ny, g_M, m, Z_s0, lambda);

    CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&iEnd);
    printf("\t[ backscatterComputeGPU\t<<<(%d,%d), (%d,%d)>>> ] \telapsed %f s\n\n",
           grid.x, grid.y, block.x, block.y,
           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart
    );

    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(sigma, g_sigma, nBytes, cudaMemcpyDeviceToHost));

    /*------------------------------调整输出图像的大小-------------------------------*/
    QueryPerformanceCounter(&iStart);

    // 可用并行归约优化！
    printf("adjusting the size of output image...\n");
    // 只有图像的宽度范围需要调整
    int miny = 1e5, maxy = -1e5;

    for (int x = 0; x < nx - 1; x++) {
        for (int y = 0; y < ny - 1; y++) {
            // Leberl 构像模型计算模拟 SAR 图像的纵坐标
            int Y = ceil(
                    (sqrt((y * m - Y_s0) * (y * m - Y_s0) + (M[x * ny + y] - Z_s0) * (M[x * ny + y] - Z_s0)) - R_0) /
                    M_s);

            if (Y < miny) {
                miny = Y;
            }
            if (Y > maxy) {
                maxy = Y;
            }
        }
    }

    int delta = 100; // 左右留空
    int imgNx = nx;
    int imgNy = (maxy - miny) + 2 * delta;
    int imgSize = imgNx * imgNy;

    QueryPerformanceCounter(&iEnd);
    printf("\t[ adjusting the size of output image ] \t\t\telapsed %f s\n\n",
           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart
    );

    /*------------------------------SAR模拟-----------------------------------*/
    float *img = (float *) malloc(imgSize * sizeof(float));
    memset(img, 0, imgSize * sizeof(float));

    // allocate device memory
    float *g_img = NULL;
    CHECK(cudaMalloc((void **) &g_img, imgSize * sizeof(float)));

    // execution configuration

    // kernel: imageSimulationGPU
    QueryPerformanceCounter(&iStart);

    imageSimulationGPU<<<grid, block>>>(g_img, imgNy, miny, delta, nx, ny, g_M, g_sigma, g_shadow, m, Y_s0, Z_s0, R_0,
                                        M_s);

    CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&iEnd);
    printf("\t[ imageSimulationGPU\t<<<(%d,%d), (%d,%d)>>> ] \telapsed %f s\n\n",
           grid.x, grid.y, block.x, block.y,
           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart
    );

    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(img, g_img, imgSize * sizeof(float), cudaMemcpyDeviceToHost));

    /*-------------------------------图像处理---------------------------------*/
    // 均值滤波
    img = imfliter(img, imgNx, imgNy);

    // 归一化为灰度图像（可用并行化优化）
    normalized(img, imgNx, imgNy);

    /*-------------------------------保存图像---------------------------------*/
    QueryPerformanceCounter(&iStart);

    saveMatrix(img, imgNx, imgNy, "img");

    QueryPerformanceCounter(&iEnd);
    printf("\t[ imageSimulationGPU\t<<<(%d,%d), (%d,%d)>>> ] \telapsed %f s\n\n",
           grid.x, grid.y, block.x, block.y,
           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart
    );

    printf("done.\n\n");

    /*--------------------------------回波模拟----------------------------------------*/
#ifdef ECHO_SIMU_GPU

    printf("echo simulating...\n");

    /*--------------------------------数据预处理--------------------------------------*/
    // 均值滤波
    sigma = imfliter(sigma, nx, ny);

    // 归一化
    normalized(sigma, nx, ny);

    // 生成数据
    int target_size = nxy;
    float *target = (float *) malloc(target_size * 4 * sizeof(float));
    memset(target, 0, target_size * 4 * sizeof(float));
    int idx = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            target[idx * 4 + 0] = j - ny / 2;
            target[idx * 4 + 1] = i - nx / 2;
            target[idx * 4 + 2] = M[i * ny + j];
            target[idx * 4 + 3] = sigma[i * ny + j] * shadow[i * ny + j];
            idx++;
        }
    }

//    saveMatrix(target, target_size, 4, "tmp");

//    float target[12] = {0, 0, 0, 2,
//                        80, 45, 0, -2,
//                        -20, -20, 0, -10};
//    int target_size = 3;


    /*--------------------------------雷达参数----------------------------------------*/
    float Kr = B / Tp;    // 调频率
    float fs = 1.6 * B;     // 采样率

    float H = 1730;     // 飞机高度
    float Ls = 200;     // 合成孔径长度
    float v = 100;      // 飞机速度
    float Lt = Ls / v;    // 合成孔径时间

    // 成像区域[Xc-X0,Xc+X0; Yc-Y0,Yc+Y0]
    // 以合成孔径中心为原点，距离向为x轴，方位向为y轴
    float Xc = 2500;
    float Yc = 0;
    float Xo = 700;
    float Yo = 512;


    for (int i = 0; i < target_size; i++) {
        target[i * 4 + 0] += Xc;
        target[i * 4 + 1] += Yc;
    }

    float Rc = sqrt(H * H + Xc * Xc);     // 中心距离
    float Ka = 2 * v * v / (Rc * lambda);   // 多普勒调频率
    float Bmax = Lt * Ka;            // 多普勒最大带宽
    float fa = ceil(3 * Bmax);       // 脉冲重复频率

    float Rmin = sqrt(H * H + (Xc - Xo) * (Xc - Xo));   // 观测场景距飞机的最近距离
    float Rmax = sqrt((Xc + Xo) * (Xc + Xo) + H * H + (Yc + Yo + Ls / 2) * (Yc + Yo + Ls / 2));  // 最远距离
    float rm = Ls + 2 * Yo;    // 雷达走过的总路程长度

    int tm_size = (rm * fa) / v;
    float *tm = (float *) malloc(tm_size * sizeof(float));                  // 慢时间（合成孔径时间+成像区域时间）
    for (int i = 0; i < tm_size; i++) {
        tm[i] = i / fa;
    }

    int tk_size = (2 * Rmax / C - 2 * Rmin / C + Tp) * fs;
    float *tk = (float *) malloc(tk_size * sizeof(float));       // 快时间（距离门内）
    for (int i = 0; i < tk_size; i++) {
        tk[i] = (2 * Rmin / C - Tp / 2) + i / fs;
    }

    float *echo_real = (float *) malloc(tm_size * tk_size * sizeof(float)); // 回波
    memset(echo_real, 0, tm_size * tk_size * sizeof(float));

    float *echo_imag = (float *) malloc(tm_size * tk_size * sizeof(float)); // 回波
    memset(echo_imag, 0, tm_size * tk_size * sizeof(float));

    float *y = (float *) malloc(tm_size * sizeof(float)); // 飞机y轴坐标
    for (int i = 0; i < tm_size; i++) {
        y[i] = -v * (rm / v) / 2 + v * tm[i];
    }

    /*--------------------------------生成回波----------------------------------------*/

    int val = target_size;
    CHECK(cudaMemcpyToSymbol(target_count, &val, sizeof(int)));

    // allocate device memory
    float *g_real = NULL;
    CHECK(cudaMalloc((void **) &g_real, tm_size * tk_size * sizeof(float)));

    float *g_imag = NULL;
    CHECK(cudaMalloc((void **) &g_imag, tm_size * tk_size * sizeof(float)));

    float *g_target = NULL;
    CHECK(cudaMalloc((void **) &g_target, target_size * 4 * sizeof(float)));
    CHECK(cudaMemcpy(g_target, target, target_size * 4 * sizeof(float), cudaMemcpyHostToDevice));

    float *g_tk = NULL;
    CHECK(cudaMalloc((void **) &g_tk, tk_size * sizeof(float)));
    CHECK(cudaMemcpy(g_tk, tk, tk_size * sizeof(float), cudaMemcpyHostToDevice));

    float *g_y = NULL;
    CHECK(cudaMalloc((void **) &g_y, tm_size * sizeof(float)));
    CHECK(cudaMemcpy(g_y, y, tm_size * sizeof(float), cudaMemcpyHostToDevice));

    // execution configuration
    grid.x = (block.x + tm_size - 1) / block.x;
    grid.y = (block.y + tk_size - 1) / block.y;

    // kernel: createEchoGPU
    QueryPerformanceCounter(&iStart);

    createEchoGPU<<<grid, block>>>(g_real, g_imag, g_target, g_tk, g_y, target_size, tk_size, H, Kr, Xc, C, Fc, PI, Tp);

    CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&iEnd);
    printf("\t[ createEchoGPU\t<<<(%d,%d), (%d,%d)>>> ] \telapsed %f s\n\n",
           grid.x, grid.y, block.x, block.y,
           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart
    );

    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(echo_real, g_real, tm_size * tk_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(echo_imag, g_imag, tm_size * tk_size * sizeof(float), cudaMemcpyDeviceToHost));

    // save
    saveMatrix(echo_real, tm_size, tk_size, "real");
    saveMatrix(echo_imag, tm_size, tk_size, "imag");
#endif

    /*-----------------------------------释放资源---------------------------*/
    free(M);
    free(shadow);
    free(sigma);
    free(img);

#ifdef ECHO_SIMU_GPU
    free(target);
    free(tm);
    free(tk);
    free(y);
    free(echo_real);
    free(echo_imag);
#endif

    CHECK(cudaFree(g_M));
    CHECK(cudaFree(g_shadow));
    CHECK(cudaFree(g_sigma));
    CHECK(cudaFree(g_img));

#ifdef ECHO_SIMU_GPU
    CHECK(cudaFree(g_target));
    CHECK(cudaFree(g_tk));
    CHECK(cudaFree(g_y));
    CHECK(cudaFree(g_real));
    CHECK(cudaFree(g_imag));
#endif
}

void CPU();

void echo_test();

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
//    echo_test();
    return 0;
}

void CPU() {
    printf("using CPU\n");

    // set timer
    LARGE_INTEGER iStart, iEnd, tc;
    QueryPerformanceFrequency(&tc);

//    /*---------------------------create a cone model----------------------*/
//    QueryPerformanceCounter(&iStart);
//    const int cone_row = 1024, cone_col = 2000;
//    int nx = cone_row, ny = cone_col;
//    int nxy = nx * ny;
//    int nBytes = nxy * sizeof(float);
//
//    float *M = (float *) malloc(nBytes);
//
//    memset(M, 0, nBytes);
//
//    for (int i = 0; i < nx; i++) {
//        for (int j = 0; j < ny; j++) {
//            for (int k = 0; k < 400; k++) {
//                if ((i - 512) * (i - 512) + (j - 700) * (j - 700) < k * k) {
//                    M[i * ny + j]++;
//                }
//            }
//        }
//    }
//    QueryPerformanceCounter(&iEnd);
//    printf("create cone elapsed %f s\n", (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart);
//
////    saveMatrix(M, nx, ny, "gpuCone");

    /*------------------------------导入数据-----------------------------------*/
    QueryPerformanceCounter(&iStart);

    int nx = 0, ny = 0;
    vector<float> DEM = load(nx, ny);
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    float *M = (float *) malloc(nBytes);

    for (int i = 0; i < nxy; i++) {
        M[i] = DEM[i];
    }

    QueryPerformanceCounter(&iEnd);
    printf("loading DEM data elapsed %f s\n", (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart);

    /*------------------------------雷达参数-----------------------------------*/

    // 雷达初始位置
    float X_s0 = 0;
//    float Y_s0 = -2000;
//    float Z_s0 = 1730;
    float Y_s0 = -5000;
    float Z_s0 = 5000;

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
    printf("shadow computing elapsed %f s\n", (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart);

    /*------------------------------后向散射系数计算---------------------------*/
    QueryPerformanceCounter(&iStart);

    float *sigma = (float *) malloc(nBytes);
    memset(sigma, 0, nBytes);

    for (int x = 0; x < nx - 1; x++) {
        for (int y = 0; y < ny - 1; y++) {
            // 小面元，取四个点：(0, 0, z1) (m, 0, z2) (m, m, z3) (0, m, z4)，
            // 最小二乘法拟合平面：ax + by + c
            // 论文：《利用 RD 模型和 DEM 数据的高分辨率机载 SAR 图像模拟》 王庆
            float z1 = M[x * ny + y];
            float z2 = M[(x + 1) * ny + y];
            float z3 = M[(x + 1) * ny + (y + 1)];
            float z4 = M[x * ny + (y + 1)];
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
            float theta = 0;
            float dividend = abs(b * sin(theta1) + cos(theta1));
            float divisor = sqrt(a * a + b * b + 1);
            if (abs(dividend - divisor) < 1e-6) { // 防止出现 -nan
                theta = 0;
            } else {
                theta = acos(dividend / divisor);
            }

            // 后向散射系数：Muhleman 半经验模型
//            sigma[x * ny + y] = 10 * log10(0.0133 * cos(theta) / pow((sin(theta) + 0.1 * cos(theta)), 3));

            // currie: 《典型地形的 SAR 回波模拟及快速实现》张吉宇 p25
            float S = m * m * sqrt(a * a + b * b + 1); // 小面元的截面积
            float avgh = (z1 + z2 + z3 + z4) / 4; // 高度平均值
            float sigma_h = sqrt((z1 - avgh) * (z1 - avgh) + (z2 - avgh) * (z2 - avgh) + (z3 - avgh) * (z3 - avgh) +
                                 (z4 - avgh) * (z4 - avgh)) / 3; // 标准差
            // 树林
            float A = 0.00054;
            float B = 0.64;
            float C = 0.02;
            float D = 0;

            float lambda = C / Fc;   // 波长

            float sigma_a = A * pow(C + theta, B) * exp(-D * lambda / (0.1 * sigma_h + lambda));
            sigma[x * ny + y] = sigma_a * S * cos(theta) * cos(theta);
        }
    }

    QueryPerformanceCounter(&iEnd);
    printf("Backscattering coefficient computing elapsed %f s\n",
           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart);

    /*------------------------------调整输出图像的大小-------------------------------*/
    QueryPerformanceCounter(&iStart);

    printf("adjusting the size of output image.\n");
    // 只有图像的宽度范围需要调整
    int miny = 1e5, maxy = -1e5;

    for (int x = 0; x < nx - 1; x++) {
        for (int y = 0; y < ny - 1; y++) {
            // Leberl 构像模型计算模拟 SAR 图像的纵坐标
            int Y = ceil(
                    (sqrt((y * m - Y_s0) * (y * m - Y_s0) + (M[x * ny + y] - Z_s0) * (M[x * ny + y] - Z_s0)) - R_0) /
                    M_s);

            if (Y < miny) {
                miny = Y;
            }
            if (Y > maxy) {
                maxy = Y;
            }
        }
    }

    int delta = 100; // 左右留空
    int imgNx = nx;
    int imgNy = (maxy - miny) + 2 * delta;
    int imgSize = imgNx * imgNy;

    QueryPerformanceCounter(&iEnd);
    printf("\t[ adjusting the size of output image ] \telapsed %f s\n",
           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart
    );

    /*------------------------------SAR模拟-----------------------------------*/
    QueryPerformanceCounter(&iStart);

    float *img = (float *) malloc(imgSize * sizeof(float));
    memset(img, 0, imgSize * sizeof(float));

    for (int x = 0; x < nx - 1; x++) {
        for (int y = 0; y < ny - 1; y++) {
            // Leberl 构像模型计算模拟 SAR 图像的纵坐标
            int Y = ceil(
                    (sqrt((y * m - Y_s0) * (y * m - Y_s0) + (M[x * ny + y] - Z_s0) * (M[x * ny + y] - Z_s0)) - R_0) /
                    M_s);

            float tmp = sigma[x * ny + y] * shadow[x * ny + y];
            img[x * imgNy + (Y - miny + delta)] += tmp;
        }
    }

    QueryPerformanceCounter(&iEnd);
    printf("SAR image simulation elapsed %f s\n", (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart);

    /*---------------------------------处理+保存--------------------------------------*/
    // 归一化为灰度图像...

    // 保存
    saveMatrix(img, imgNx, imgNy, "img");

    free(M);
    free(shadow);
    free(sigma);
    free(img);
}

void echo_test() {
    /*--------------------------------回波模拟----------------------------------------*/
    printf("echo simulating...\n");

    // set timer
    LARGE_INTEGER iStart, iEnd, tc;
    QueryPerformanceFrequency(&tc);

    /*--------------------------------数据预处理--------------------------------------*/

    float target[12] = {0, 0, 0, 2,
                        80, 45, 0, -2,
                        -20, -20, 0, -10};
    int target_size = 3;


    /*--------------------------------雷达参数----------------------------------------*/
    float Kr = B / Tp;    // 调频率
    float fs = 1.6 * B;     // 采样率

    float H = 1730;     // 飞机高度
    float Ls = 200;     // 合成孔径长度
    float v = 100;      // 飞机速度
    float Lt = Ls / v;    // 合成孔径时间

    // 成像区域[Xc-X0,Xc+X0; Yc-Y0,Yc+Y0]
    // 以合成孔径中心为原点，距离向为x轴，方位向为y轴
    float Xc = 2000;
    float Yc = 0;
    float Xo = 100;
    float Yo = 100;


    for (int i = 0; i < target_size; i++) {
        target[i * 4 + 0] += Xc;
        target[i * 4 + 1] += Yc;
    }

    float Rc = sqrt(H * H + Xc * Xc);     // 中心距离
    float Ka = 2 * v * v / (Rc * lambda);   // 多普勒调频率
    float Bmax = Lt * Ka;            // 多普勒最大带宽
    float fa = ceil(3 * Bmax);       // 脉冲重复频率

    float Rmin = sqrt(H * H + (Xc - Xo) * (Xc - Xo));   // 观测场景距飞机的最近距离
    float Rmax = sqrt((Xc + Xo) * (Xc + Xo) + H * H + (Yc + Yo + Ls / 2) * (Yc + Yo + Ls / 2));  // 最远距离
    float rm = Ls + 2 * Yo;    // 雷达走过的总路程长度

    int tm_size = (rm * fa) / v;
    float *tm = (float *) malloc(tm_size * sizeof(float));                  // 慢时间（合成孔径时间+成像区域时间）
    for (int i = 0; i < tm_size; i++) {
        tm[i] = i / fa;
    }

    int tk_size = (2 * Rmax / C - 2 * Rmin / C + Tp) * fs;
    float *tk = (float *) malloc(tk_size * sizeof(float));       // 快时间（距离门内）
    for (int i = 0; i < tk_size; i++) {
        tk[i] = (2 * Rmin / C - Tp / 2) + i / fs;
    }

    float *echo_real = (float *) malloc(tm_size * tk_size * sizeof(float)); // 回波
    memset(echo_real, 0, tm_size * tk_size * sizeof(float));

    float *echo_imag = (float *) malloc(tm_size * tk_size * sizeof(float)); // 回波
    memset(echo_imag, 0, tm_size * tk_size * sizeof(float));

    float *y = (float *) malloc(tm_size * sizeof(float)); // 飞机y轴坐标
    for (int i = 0; i < tm_size; i++) {
        y[i] = -v * (rm / v) / 2 + v * tm[i];
    }

    for (int i = 0; i < tm_size; i++) { // 慢时间轴
        for (int j = 0; j < tk_size; j++) { // 快时间轴
            for (int k = 0; k < target_size; k++) { // 目标数
                float Xs = target[k * 4 + 0];
                float Ys = target[k * 4 + 1];
                float Zs = target[k * 4 + 2];
                float sigma = target[k * 4 + 3];

                if (sigma == 0) continue;

                float Rtm = sqrt(pow(Xs, 2) + pow(Ys - y[i], 2) + pow(Zs - H, 2));

                if ((abs(target[k * 4 + 1] - y[i]) / Xc) < 0.01) { // 假定波束宽度是0.02°
                    float rec = 0;
                    if (abs(tk[j] - 2 * Rtm / C) < Tp / 2) {
                        rec = 1;
                    }
                    float exp_real = cos(2 * PI * Fc * (tk[j] - 2 * Rtm / C) + PI * Kr * pow(tk[j] - 2 * Rtm / C, 2));
                    float exp_imag = sin(2 * PI * Fc * (tk[j] - 2 * Rtm / C) + PI * Kr * pow(tk[j] - 2 * Rtm / C, 2));
                    echo_real[i * tk_size + j] += sigma * rec * exp_real;
                    echo_imag[i * tk_size + j] += sigma * rec * exp_imag;
                }
            }
        }
    }

//    saveMatrix(echo_real, tm_size, tk_size, "real");
//    saveMatrix(echo_imag, tm_size, tk_size, "imag");

    // GPU

    float *test_real = (float *) malloc(tm_size * tk_size * sizeof(float)); // 回波
    memset(test_real, 0, tm_size * tk_size * sizeof(float));

    float *test_imag = (float *) malloc(tm_size * tk_size * sizeof(float)); // 回波
    memset(test_imag, 0, tm_size * tk_size * sizeof(float));

    // allocate device memory
    float *g_real = NULL;
    CHECK(cudaMalloc((void **) &g_real, tm_size * tk_size * sizeof(float)));

    float *g_imag = NULL;
    CHECK(cudaMalloc((void **) &g_imag, tm_size * tk_size * sizeof(float)));

    float *g_target = NULL;
    CHECK(cudaMalloc((void **) &g_target, target_size * 4 * sizeof(float)));
    CHECK(cudaMemcpy(g_target, target, target_size * 4 * sizeof(float), cudaMemcpyHostToDevice));

    float *g_tk = NULL;
    CHECK(cudaMalloc((void **) &g_tk, tk_size * sizeof(float)));
    CHECK(cudaMemcpy(g_tk, tk, tk_size * sizeof(float), cudaMemcpyHostToDevice));

    float *g_y = NULL;
    CHECK(cudaMalloc((void **) &g_y, tm_size * sizeof(float)));
    CHECK(cudaMemcpy(g_y, y, tm_size * sizeof(float), cudaMemcpyHostToDevice));

    // execution configuration
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((block.x + tm_size - 1) / block.x, (block.y + tk_size - 1) / block.x);

    // kernel: createEchoGPU
    QueryPerformanceCounter(&iStart);

    createEchoGPU<<<grid, block>>>(g_real, g_imag, g_target, g_tk, g_y, target_size, tk_size, H, Kr, Xc, C, Fc, PI, Tp);

    CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&iEnd);
    printf("\t[ createEchoGPU\t<<<(%d,%d), (%d,%d)>>> ] \telapsed %f s\n\n",
           grid.x, grid.y, block.x, block.y,
           (float) (iEnd.QuadPart - iStart.QuadPart) / (float) tc.QuadPart
    );

    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(test_real, g_real, tm_size * tk_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(test_imag, g_imag, tm_size * tk_size * sizeof(float), cudaMemcpyDeviceToHost));

    checkResultGPU(echo_real, test_real, tm_size * tk_size);
    checkResultGPU(echo_imag, test_imag, tm_size * tk_size);

    saveMatrix(test_real, tm_size, tk_size, "real");
    saveMatrix(test_imag, tm_size, tk_size, "imag");
}
