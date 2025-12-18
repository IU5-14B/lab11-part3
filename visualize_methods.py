# coding: utf-8
"""
visualize_methods.py
====================

Этот скрипт сравнивает время работы трёх способов решения систем линейных
уравнений: метода исключения Гаусса, метода LU‑разложения (по схеме
Doolittle) и метода решения через обратную матрицу A^{-1}.  Для каждого
размера матрицы n генерируется случайная невырожденная матрица и вектор
свободных членов, затем измеряется среднее время работы каждого
алгоритма на нескольких прогонах.  Результат выводится в виде графика.

Запуск:

    python3 scripts/visualize_methods.py

Файл сохраняет график в текущую директорию под именем
``comparison.png`` и также отображает его на экране (если это
поддерживается окружением).

Примечание: алгоритмы реализованы на Python и предназначены для
показательных измерений.  Для крупных n они работают сравнительно
медленно, так как асимптотическая сложность всех трёх методов
O(n^3).  Однако полученные относительные времена позволяют наглядно
сравнить подходы.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решает СЛАУ Ax=b методом Гаусса с частичным выбором главного элемента.

    Аргументы:
        A: квадратная матрица коэффициентов (n×n).
        b: столбец свободных членов (длина n).

    Возвращает:
        Вектор-столбец решения x (длина n).
    """
    # Преобразуем к типу float для предотвращения переполнения целых
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = A.shape[0]
    # Прямой ход
    for col in range(n):
        # Поиск главного элемента
        pivot = col
        max_abs = abs(A[pivot, col])
        for row in range(col + 1, n):
            val = abs(A[row, col])
            if val > max_abs:
                max_abs = val
                pivot = row
        if max_abs < EPS:
            raise ValueError("Матрица вырождена или близка к вырожденной")
        # Перестановка строк
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
            b[[col, pivot]] = b[[pivot, col]]
        # Обнуляем элементы ниже ведущего
        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row, col] = 0.0
            A[row, col + 1 :] -= factor * A[col, col + 1 :]
            b[row] -= factor * b[col]
    # Обратная подстановка
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = np.dot(A[i, i + 1 :], x[i + 1 :])
        x[i] = (b[i] - sum_ax) / A[i, i]
    return x


def lu_decomposition(A: np.ndarray):
    """Выполняет LU‑разложение матрицы A по схеме Doolittle.

    Возвращает кортеж (L, U).  Если разложение невозможно (нулевой
    ведущий элемент), возбуждается исключение ValueError.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        L[i, i] = 1.0
    for i in range(n):
        # Вычисляем элементы матрицы U
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - s
        if abs(U[i, i]) < EPS:
            raise ValueError("Нулевой ведущий элемент в LU‑разложении")
        # Вычисляем элементы матрицы L
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += L[j, k] * U[k, i]
            L[j, i] = (A[j, i] - s) / U[i, i]
    return L, U


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решает систему Ly=b методом прямой подстановки (L — нижнетреугольная).
    """
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]
    return y


def back_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Решает систему Ux=y методом обратной подстановки (U — верхнетреугольная).
    """
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]
    return x


def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решает СЛАУ Ax=b, используя LU‑разложение.

    Если разложение невозможно, возбуждается исключение ValueError.
    """
    L, U = lu_decomposition(A.astype(float))
    y = forward_substitution(L, b.astype(float))
    x = back_substitution(U, y)
    return x


def invert_matrix(A: np.ndarray) -> np.ndarray:
    """Вычисляет обратную матрицу A^{-1} методом Гаусса–Жордана.

    Если матрица вырождена, возбуждается исключение ValueError.
    """
    n = A.shape[0]
    # Формируем расширенную матрицу [A | I]
    aug = np.hstack((A.astype(float).copy(), np.eye(n)))
    for col in range(n):
        # Выбор главного элемента
        pivot = col + np.argmax(np.abs(aug[col:, col]))
        if abs(aug[pivot, col]) < EPS:
            raise ValueError("Матрица вырождена или близка к вырожденной")
        # Переставляем строки, чтобы ведущий элемент оказался на диагонали
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        # Нормализуем строку с ведущим элементом
        div = aug[col, col]
        aug[col] = aug[col] / div
        # Вычитаем из остальных строк, чтобы обнулить столбец
        for row in range(n):
            if row == col:
                continue
            factor = aug[row, col]
            aug[row] -= factor * aug[col]
    # Правая часть расширенной матрицы теперь содержит обратную
    invA = aug[:, n:]
    return invA


def solve_via_inverse(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решает СЛАУ Ax=b, умножая A^{-1} на b."""
    invA = invert_matrix(A)
    return invA.dot(b)


def measure_execution_times(min_n: int = 10, max_n: int = 60, step: int = 10, trials: int = 3):
    """Измеряет среднее время работы каждого метода для разных размеров n.

    Возвращает три списка: sizes, times_gauss, times_lu, times_inv.
    """
    sizes = list(range(min_n, max_n + 1, step))
    times_gauss = []
    times_lu = []
    times_inv = []
    for n in sizes:
        total_gauss = 0.0
        total_lu = 0.0
        total_inv = 0.0
        for _ in range(trials):
            # Генерируем случайную матрицу и вектор, добавляя диагональ для устойчивости
            A = np.random.rand(n, n) + n * np.eye(n)
            b = np.random.rand(n)
            # Gaussian elimination
            start = time.perf_counter()
            gaussian_elimination(A.copy(), b.copy())
            total_gauss += time.perf_counter() - start
            # LU decomposition
            start = time.perf_counter()
            solve_lu(A.copy(), b.copy())
            total_lu += time.perf_counter() - start
            # Inverse method
            start = time.perf_counter()
            solve_via_inverse(A.copy(), b.copy())
            total_inv += time.perf_counter() - start
        times_gauss.append(total_gauss / trials)
        times_lu.append(total_lu / trials)
        times_inv.append(total_inv / trials)
    return sizes, times_gauss, times_lu, times_inv


def main():
    sizes, times_gauss, times_lu, times_inv = measure_execution_times()
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times_gauss, marker='o', label='Метод Гаусса')
    plt.plot(sizes, times_lu, marker='s', label='LU‑разложение')
    plt.plot(sizes, times_inv, marker='^', label='Обратная матрица')
    plt.xlabel('Размер системы n')
    plt.ylabel('Среднее время (сек)')
    plt.title('Сравнение времени работы методов решения СЛАУ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Сохраняем и отображаем график
    plt.savefig('comparison.png')
    try:
        plt.show()
    except Exception:
        # В неинтерактивных окружениях plt.show() может ничего не выводить
        pass


if __name__ == '__main__':
    main()