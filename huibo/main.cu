#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <cstring>
#include <windows.h>
#include <math.h>

#include "save.h"

const float PI = 3.14159265358979323846;    // pi

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

int nextpow2(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void echoCPU() {
    printf("using CPU\n");

    /*------------------------------参数设置-------------------------*/
    float c = 3e8;                          // 光速
    float fc = 3e9;                         // 载波频率（Hz)
    float lambda = c / fc;                  // 波长0.1（m）
    float Ralpha = 1.2;
    float Aalpha = 1.3;                     // 距离向、方位向过采样率

    /*-------------------------观测场景相关参数------------------------*/
    float H = 8000;                       // 平台运行高度
    float Yc = 6000;                      // 场景中心线长度
    float R0 = sqrt(H * H + Yc * Yc);
    float Xmin = -200;                    // [Xmin Xmax]
    float Xmax = 200;                     // 在合成孔径长度Ls基础上，额外方位向范围
    float Yw = 300;

    /*---------------------------天线孔径设置-------------------------*/
    float La = 1.5;
    float theta = 0.886 * lambda / La;
    float Ls = R0 * theta;

    /*---------------------------慢时间参数设置------------------------*/
    float v = 150;                              // 平台运行速度
    float Ts = Ls / v;
    float Xwid = Ls + Xmax - Xmin;              // 慢时间域时间窗长度
    float Twid = Xwid / v;
    float Ka = 2 * v * v / lambda / R0;          // 方位向调频率
    float Ba = abs(Ka * Ts);                // 方位向带宽
    float PRF = Aalpha * Ba;
    float PRT = 1 / PRF;
    float dx = PRT;                             // 方位向采样间隔
    int N = ceil(Twid / dx);
    N = nextpow2(N);                            // 提高fft效率

    float *x = (float *) malloc(N * sizeof(float));             // 慢时间域时间序列
    float *X = (float *) malloc(N * sizeof(float));             // 慢时间域时间序列对应方位向距离
    float begin = (Xmin - Ls / 2) / v;
    float end = (Xmax + Ls / 2) / v;
    float step = (end - begin) / (N - 1);
    x[0] = begin;
    X[0] = x[0] * v;
    for (int i = 1; i < N; i++) {
        x[i] = x[i - 1] + step;
        X[i] = x[i] * v;
    }

    // 更新
    PRT = Twid / N;
    PRF = 1 / PRT;
    dx = PRT;

    /*---------------------------快时间参数设置------------------------*/
    float Tr = 5e-6;                            // 脉冲宽度
    float Br = 100e6;                           // 带宽100MHz
    float Kr = Br / Tr;                         // 调频率
    float Fs = Ralpha * Br;                     // 距离向采样率
    float dt = 1 / Fs;
    float Rmin = sqrt((Yc - Yw) * (Yc - Yw) + H * H);
    float Rmax = sqrt((Yc + Yw) * (Yc + Yw) + H * H);
    float Rm = Rmax - Rmin + c * Tr / 2;            // 斜距测绘带宽
    int M = ceil(2 * Rm / c / dt);            // 采样点数
    M = nextpow2(M);

    float *t = (float *) malloc(M * sizeof(float)); // 快时间域时间序列
    float *r = (float *) malloc(M * sizeof(float));
    begin = 2 * Rmin / c - Tr / 2;
    end = 2 * Rmax / c + Tr / 2;
    step = (end - begin) / (M - 1);
    t[0] = begin;
    r[0] = t[0] * c / 2;
    for (int i = 1; i < M; i++) {
        t[i] = t[i - 1] + step;
        r[i] = t[i] * c / 2;
    }

    // 更新
    dt = (2 * Rmax / c + Tr - 2 * Rmin / c) / M;
    Fs = 1 / dt;

    /*-----------------------------目标参数设置------------------------*/
    float Ntarget = 3;
//    float* Ptarget = (float *) malloc(Ntarget * 3 * sizeof(float));
    float Ptarget[3 * 3] = {0, R0, 1,
                            50, R0 + 100, 0.8,
                            100, R0 + 100, 0.8};

    /*---------------------------生成 SAR 回波------------------------*/
    float *s0_real = (float *) malloc(N * M * sizeof(float));
    float *s0_imag = (float *) malloc(N * M * sizeof(float));
    memset(s0_real, 0, N * M * sizeof(float));
    memset(s0_imag, 0, N * M * sizeof(float));

    for (int k = 0; k < Ntarget; k++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                float sigma = Ptarget[k * 3 + 2];
                float R = sqrt(Ptarget[k * 3 + 1] * Ptarget[k * 3 + 1] +
                               (X[i] - Ptarget[k * 3 + 0]) * (X[i] - Ptarget[k * 3 + 0]));
                float delay = 2 * R / c;
                float Delay = t[j] - delay;
                float Phase = PI * Kr * Delay * Delay - (4 * PI * fc * R) / c;
                if (abs(Delay / Tr) < 0.5 && abs((X[i] - Ptarget[k * 3 + 0]) / Ls) < 0.5) {
                    s0_real[i * M + j] += sigma * cos(Phase);
                    s0_imag[i * M + j] += sigma * sin(Phase);
                }
            }
        }
    }

    /*---------------------------保存 SAR 回波------------------------*/
    saveMatrix(s0_real, N, M, "real");
    saveMatrix(s0_imag, N, M, "imag");

    /*-----------------------------释放资源--------------------------*/
    free(x);
    free(X);
    free(t);
    free(r);
//    free(Ptarget);
    free(s0_real);
    free(s0_imag);

#ifdef TEST
#endif

}

int main() {
    echoCPU();
    return 0;
}