#include "matrix.h"
#include <iomanip>

Matrix::Matrix() : m_rows(0), m_cols(0) {}

Matrix::Matrix(int rows, int cols, double value)
    : m_rows(rows), m_cols(cols), a(rows, std::vector<double>(cols, value)) {}

int Matrix::rows() const { return m_rows; }
int Matrix::cols() const { return m_cols; }

double& Matrix::at(int r, int c) { return a[r][c]; }
double  Matrix::at(int r, int c) const { return a[r][c]; }

Matrix Matrix::identity(int n) {
    Matrix I(n, n, 0.0);
    for (int i = 0; i < n; ++i) I.at(i, i) = 1.0;
    return I;
}

void Matrix::print(const std::string& title) const {
    if (!title.empty()) {
        std::cout << title << "\n";
    }
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < m_rows; ++i) {
        for (int j = 0; j < m_cols; ++j) {
            std::cout << std::setw(12) << a[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
