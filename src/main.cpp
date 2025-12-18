#include <iostream>
#include <fstream>
#include <filesystem>

#include "matrix.h"
#include "linear_solvers.h"

// Сохраняем псевдокод стадии обратной подстановки в файл
static void saveBackSubstitutionPseudocode(const std::string& filePath) {
    std::ofstream out(filePath);
    if (!out.is_open()) {
        std::cerr << "Не удалось открыть файл для записи: " << filePath << "\n";
        return;
    }

    out << "Псевдокод: обратная подстановка (метод исключения Гаусса)\n";
    out << "--------------------------------------------------------\n\n";
    out << "Вход: верхнетреугольная матрица U (n x n), вектор y (n)\n";
    out << "Выход: вектор решения x (n)\n\n";

    out << "x[n-1] = y[n-1] / U[n-1][n-1]\n";
    out << "для i = n-2 ... 0:\n";
    out << "    sum = 0\n";
    out << "    для j = i+1 ... n-1:\n";
    out << "        sum = sum + U[i][j] * x[j]\n";
    out << "    x[i] = (y[i] - sum) / U[i][i]\n\n";

    out << "Доказательство сложности Θ(n^2):\n";
    out << "Внешний цикл выполняется (n-1) раз.\n";
    out << "Внутренний цикл выполняется: (n-1) + (n-2) + ... + 1 = n(n-1)/2.\n";
    out << "Следовательно, число операций ~ n^2/2, то есть Θ(n^2).\n";

    out.close();
}

int main() {
    std::cout << "Лабораторная работа 11, часть 3\n";
    std::cout << "Решение системы линейных уравнений тремя способами\n\n";

    // Папка для результатов
    std::filesystem::create_directories("results");

    // Сохраняем псевдокод в файл (задание 5)
    const std::string pseudoFile = "results/pseudocode_back_substitution.txt";
    saveBackSubstitutionPseudocode(pseudoFile);
    std::cout << "Псевдокод обратной подстановки сохранён в файл: " << pseudoFile << "\n\n";

    // Система из задания:
    // x1 + x2 + x3 = 2
    // 2x1 + x2 + x3 = 3
    // x1 - x2 + 3x3 = 8

    Matrix A(3, 3, 0.0);
    A.at(0, 0) = 1;  A.at(0, 1) = 1;   A.at(0, 2) = 1;
    A.at(1, 0) = 2;  A.at(1, 1) = 1;   A.at(1, 2) = 1;
    A.at(2, 0) = 1;  A.at(2, 1) = -1;  A.at(2, 2) = 3;

    std::vector<double> b = {2, 3, 8};

    A.print("Матрица коэффициентов A:");
    std::cout << "Вектор свободных членов b:\n";
    printVector(b);

    // 1) Гаусс
    std::vector<double> x_gauss = solveGaussian(A, b);
    if (!x_gauss.empty()) {
        printVector(x_gauss, "Решение методом исключения Гаусса:");
    }

    // 2) LU
    std::vector<double> x_lu = solveLU(A, b);
    if (!x_lu.empty()) {
        printVector(x_lu, "Решение методом LU-разложения:");
    }

    // 3) Через A^{-1} * b
    Matrix invA;
    if (invertMatrix(A, invA)) {
        invA.print("Обратная матрица A^{-1}:");
        std::vector<double> x_inv = multiply(invA, b);
        printVector(x_inv, "Решение через обратную матрицу (A^{-1} * b):");
    } else {
        std::cout << "Обратная матрица: вычислить не удалось (матрица вырождена)\n";
    }

    return 0;
}
