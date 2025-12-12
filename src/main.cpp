#include <iostream>
#include "matrix.h"
#include "linear_solvers.h"

int main() {
    std::cout << "lab11-part3: solving system in 3 ways\n\n";

    // Система из задания:
    // x1 + x2 + x3 = 2
    // 2x1 + x2 + x3 = 3
    // x1 - x2 + 3x3 = 8

    Matrix A(3, 3, 0.0);
    A.at(0, 0) = 1;  A.at(0, 1) = 1;  A.at(0, 2) = 1;
    A.at(1, 0) = 2;  A.at(1, 1) = 1;  A.at(1, 2) = 1;
    A.at(2, 0) = 1;  A.at(2, 1) = -1; A.at(2, 2) = 3;

    std::vector<double> b = {2, 3, 8};

    A.print("Matrix A:");
    std::cout << "Vector b:\n";
    printVector(b);

    // 1) Гаусс
    std::vector<double> x_gauss = solveGaussian(A, b);
    if (!x_gauss.empty()) {
        printVector(x_gauss, "Solution by Gaussian elimination:");
    }

    // 2) LU
    std::vector<double> x_lu = solveLU(A, b);
    if (!x_lu.empty()) {
        printVector(x_lu, "Solution by LU decomposition:");
    }

    // 3) Через A^{-1} * b
    Matrix invA;
    if (invertMatrix(A, invA)) {
        invA.print("Inverse matrix A^{-1}:");
        std::vector<double> x_inv = multiply(invA, b);
        printVector(x_inv, "Solution by inverse matrix (A^{-1} * b):");
    } else {
        std::cout << "Inverse matrix: failed (singular matrix)\n";
    }

    return 0;
}
