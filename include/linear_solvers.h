#pragma once
#include "matrix.h"
#include <vector>

// Решение СЛАУ Ax=b методом Гаусса (с частичным выбором главного элемента)
std::vector<double> solveGaussian(Matrix A, std::vector<double> b);

// LU-разложение (Doolittle): A = L*U, где диагональ L = 1
// Возвращает true, если разложение удалось (нет нулевого ведущего элемента)
bool luDecompose(const Matrix& A, Matrix& L, Matrix& U);

// Решение через LU: сначала Ly=b, потом Ux=y
std::vector<double> solveLU(const Matrix& A, const std::vector<double>& b);

// Обращение матрицы методом Гаусса–Жордана. Возвращает true, если удалось.
bool invertMatrix(const Matrix& A, Matrix& invA);

// Умножение матрицы на вектор
std::vector<double> multiply(const Matrix& A, const std::vector<double>& x);

// Печать вектора
void printVector(const std::vector<double>& v, const std::string& title = "");
