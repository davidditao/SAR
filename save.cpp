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
    } else if (name == "tmp1") {
        ofs.open(FILETMP1, ofstream::out);
    } else if (name == "tmp2") {
        ofs.open(FILETMP2, ofstream::out);
    } else if (name == "real") {
        ofs.open(FILEREAL, ofstream::out);
    } else if (name == "imag") {
        ofs.open(FILEIMAG, ofstream::out);
    } else {
        ofs.open(FILETMP, ofstream::out);
    }

    int idx = 0;
    printf("saving %s(%d, %d)...\n", name.c_str(), nx, ny);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofs << std::fixed << std::setprecision(6) << matrix[idx++];
            if (j != ny - 1) {
                ofs << ",";
            }
        }
        ofs << endl;
    }
    ofs.close();
}

vector<float> load(int &nx, int &ny) {
    printf("loading DEM data.\n");
    ifstream ifs;
    ifs.open(FILEDEM, ios::in);

    if (!ifs) {
        printf("open file error!\n");
        exit(0);
    }

    nx = 0, ny = 0;
    vector<float> res;
    string line;
    while (getline(ifs, line)) {
        nx++;
        string num;
        stringstream ss(line);

        while (getline(ss, num, ',')) {
            res.push_back(atof(num.c_str()));
        }
    }
    ny = (int) res.size() / nx;

    ifs.close();

    return res;
}