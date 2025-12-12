#include "linear_solvers.h"
#include <cmath>
#include <iostream>
#include <iomanip>

static const double EPS = 1e-12;

void printVector(const std::vector<double>& v, const std::string& title) {
    if (!title.empty()) std::cout << title << "\n";
    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << "x" << (i + 1) << " = " << v[i] << "\n";
    }
    std::cout << "\n";
}

std::vector<double> multiply(const Matrix& A, const std::vector<double>& x) {
    std::vector<double> res(A.rows(), 0.0);
    for (int i = 0; i < A.rows(); ++i) {
        double sum = 0.0;
        for (int j = 0; j < A.cols(); ++j) {
            sum += A.at(i, j) * x[j];
        }
        res[i] = sum;
    }
    return res;
}

std::vector<double> solveGaussian(Matrix A, std::vector<double> b) {
    int n = A.rows();

    // Прямой ход с частичным выбором главного элемента
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double maxAbs = std::fabs(A.at(col, col));
        for (int row = col + 1; row < n; ++row) {
            double val = std::fabs(A.at(row, col));
            if (val > maxAbs) {
                maxAbs = val;
                pivot = row;
            }
        }

        if (maxAbs < EPS) {
            std::cerr << "Метод Гаусса: матрица вырождена или близка к вырожденной.\n";
            return {};
        }

        if (pivot != col) {
            for (int j = 0; j < n; ++j) {
                std::swap(A.at(col, j), A.at(pivot, j));
            }
            std::swap(b[col], b[pivot]);
        }

        for (int row = col + 1; row < n; ++row) {
            double factor = A.at(row, col) / A.at(col, col);
            A.at(row, col) = 0.0;

            for (int j = col + 1; j < n; ++j) {
                A.at(row, j) -= factor * A.at(col, j);
            }
            b[row] -= factor * b[col];
        }
    }

    // Обратная подстановка
    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += A.at(i, j) * x[j];
        }
        x[i] = (b[i] - sum) / A.at(i, i);
    }

    return x;
}

bool luDecompose(const Matrix& A, Matrix& L, Matrix& U) {
    int n = A.rows();
    L = Matrix(n, n, 0.0);
    U = Matrix(n, n, 0.0);

    for (int i = 0; i < n; ++i) {
        L.at(i, i) = 1.0;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += L.at(i, k) * U.at(k, j);
            }
            U.at(i, j) = A.at(i, j) - sum;
        }

        if (std::fabs(U.at(i, i)) < EPS) {
            return false;
        }

        for (int j = i + 1; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += L.at(j, k) * U.at(k, i);
            }
            L.at(j, i) = (A.at(j, i) - sum) / U.at(i, i);
        }
    }

    return true;
}

static std::vector<double> forwardSubstitution(const Matrix& L, const std::vector<double>& b) {
    int n = L.rows();
    std::vector<double> y(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L.at(i, j) * y[j];
        }
        y[i] = (b[i] - sum) / L.at(i, i);
    }
    return y;
}

static std::vector<double> backSubstitution(const Matrix& U, const std::vector<double>& y) {
    int n = U.rows();
    std::vector<double> x(n, 0.0);

    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U.at(i, j) * x[j];
        }
        x[i] = (y[i] - sum) / U.at(i, i);
    }
    return x;
}

std::vector<double> solveLU(const Matrix& A, const std::vector<double>& b) {
    Matrix L, U;
    if (!luDecompose(A, L, U)) {
        std::cerr << "LU-разложение: невозможно выполнить разложение (нулевой ведущий элемент).\n";
        return {};
    }

    std::vector<double> y = forwardSubstitution(L, b);
    std::vector<double> x = backSubstitution(U, y);
    return x;
}

bool invertMatrix(const Matrix& A, Matrix& invA) {
    int n = A.rows();
    Matrix aug(n, 2 * n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug.at(i, j) = A.at(i, j);
        }
        for (int j = 0; j < n; ++j) {
            aug.at(i, n + j) = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double maxAbs = std::fabs(aug.at(col, col));
        for (int row = col + 1; row < n; ++row) {
            double val = std::fabs(aug.at(row, col));
            if (val > maxAbs) {
                maxAbs = val;
                pivot = row;
            }
        }

        if (maxAbs < EPS) return false;

        if (pivot != col) {
            for (int j = 0; j < 2 * n; ++j) {
                std::swap(aug.at(col, j), aug.at(pivot, j));
            }
        }

        double div = aug.at(col, col);
        for (int j = 0; j < 2 * n; ++j) {
            aug.at(col, j) /= div;
        }

        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double factor = aug.at(row, col);
            for (int j = 0; j < 2 * n; ++j) {
                aug.at(row, j) -= factor * aug.at(col, j);
            }
        }
    }

    invA = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            invA.at(i, j) = aug.at(i, n + j);
        }
    }
    return true;
}
