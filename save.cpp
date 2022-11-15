#include "save.h"

void saveMatrix(float **matrix, int row, int col) {
    ofstream ofs;
    ofs.open(FILENAME, ofstream::out);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            ofs << matrix[i][j] << ",";
        }
        ofs << endl;
    }
    ofs.close();
}


void saveMatrix(float *matrix, int nx, int ny, char *name) {
    ofstream ofs;
    ofs.open(FILENAME, ofstream::out);
    int idx = 0;
    printf("saving %s(%d, %d)...\n", name, nx, ny);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofs << matrix[idx++] << ",";
        }
        ofs << endl;
    }
    ofs.close();
}
