#include "save.h"

void saveArray(float *array, int len) {
    printf("saving...");
    ofstream ofs;
    ofs.open(FILETMP, ofstream::out);
    for (int i = 0; i < len; i++) {
        ofs << array[i] << ",";
    }
    ofs << endl;
    ofs.close();
}

void saveMatrix(float **matrix, int row, int col) {
    printf("saving...");
    ofstream ofs;
    ofs.open(FILETMP, ofstream::out);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            ofs << matrix[i][j] << ",";
        }
        ofs << endl;
    }
    ofs.close();
}


void saveMatrix(float *matrix, int nx, int ny, string name) {
    ofstream ofs;
    if (name == "img") {
        ofs.open(FILEIMG, ofstream::out);
    } else if (name == "echo") {
        ofs.open(FILEECHO, ofstream::out);
    } else {
        ofs.open(FILETMP, ofstream::out);
    }

    int idx = 0;
    printf("saving %s(%d, %d)...\n", name.c_str(), nx, ny);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofs << matrix[idx++] << ",";
        }
        ofs << endl;
    }
    ofs.close();
}
