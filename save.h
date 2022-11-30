#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <fstream>
#include <iomanip>

using namespace std;

#define FILEDEM     "dem.csv"

#define FILEIMG     "img.csv"

#define FILEECHO    "echo.csv"
#define FILEREAL    "echo_real.csv"
#define FILEIMAG    "echo_imag.csv"

#define FILETMP     "tmp.csv"
#define FILETMP1    "tmp1.csv"
#define FILETMP2    "tmp2.csv"

void saveArray(float *array, int len);
void saveMatrix(float **matrix, int row, int col);
void saveMatrix(float *matrix, int nx, int ny, string name);

vector<float> load(int &nx, int &ny);
