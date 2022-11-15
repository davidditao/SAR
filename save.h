#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <fstream>

using namespace std;

#define FILENAME		"E:\\CUDA\\project\\20221110\\20221110\\sar.csv"


void saveMatrix(float **matrix, int row, int col);
void saveMatrix(float *matrix, int nx, int ny, char *name);


