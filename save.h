#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <fstream>

using namespace std;

#define FILEIMG     "E:\\CUDA\\project\\20221110\\20221110\\img.csv"
#define FILEECHO    "E:\\CUDA\\project\\20221110\\20221110\\echo.csv"
#define FILETMP     "E:\\CUDA\\project\\20221110\\20221110\\tmp.csv"

void saveArray(float *array, int len);
void saveMatrix(float **matrix, int row, int col);
void saveMatrix(float *matrix, int nx, int ny, string name);


