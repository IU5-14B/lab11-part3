#pragma once
#include <vector>
#include <iostream>

class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols, double value = 0.0);

    int rows() const;
    int cols() const;

    double& at(int r, int c);
    double  at(int r, int c) const;

    static Matrix identity(int n);

    void print(const std::string& title = "") const;

private:
    int m_rows;
    int m_cols;
    std::vector<std::vector<double>> a;
};
