#include <stdio.h>  // Описания функций ввода-вывода
#include <math.h>   // Описания математических функций
#include <stdlib.h> // Описания функций malloc и free
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>

// Прототип функции приведения матрицы к ступенчатому виду.
// Функция возвращает ранг матрицы
int gaussMethod(
    int m,          // Число строк матрицы
    int n,          // Число столбцов матрицы
    double *a,      // Адрес массива элементов матрицы
    double eps      // Точность вычислений
);

int main() {
    // Переменные для измерения времени
    int num = 20;
    double start, average_time;
    // Переменные для работы с матрицами
    int m, n, i, j, rank;
    double *a;
    double eps, det;

    close(0);
    open("big_matrix_1.txt", O_RDONLY);

    printf("Введите размеры матрицы m, n: ");
    scanf("%d%d", &m, &n);
    printf("\n");

    // Захватываем память под элементы матрицы
    a = (double *) malloc(m * n * sizeof(double));

    printf("Введите элементы матрицы:\n");
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            // Вводим элемент с индексами i, j
            scanf("%lf", &(a[i*n + j]));
        }
    }
    printf("\n");

    printf("Введите точность вычислений eps: ");
    scanf("%lf", &eps);
    printf("\n");

    printf("==================================\n");
    printf("Тест функции... Число запусков: %d\n", num);
    for (int i = 0; i < num; i++) {
        // Вызываем метод Гаусса
        start = omp_get_wtime();
        rank = gaussMethod(m, n, a, eps);
        average_time += omp_get_wtime() - start;
    }
    printf("Общее время выполнения: %f\n", average_time);
    printf("Среднее время выполнения: %f\n", average_time / num);
    printf("==================================\n");

    // Печатаем ступенчатую матрицу
    // printf("Ступенчатый вид матрицы:\n");
    // for (i = 0; i < m; ++i) {
    //     // Печатаем i-ю строку матрицы
    //     for (j = 0; j < n; ++j) {
    //         printf(         // Формат %10.3lf означает 10
    //             "%10.3lf ", // позиций на печать числа,
    //             a[i*n + j]  // 3 знака после точки
    //         );
    //     }
    //     printf("\n");   // Перевести строку
    // }

    // Печатаем ранг матрицы
    printf("Ранг матрицы = %d\n", rank);

    if (m == n) {
        // Для квадратной матрицы вычисляем и печатаем
        //     ее определитель
        det = 1.0;
        for (i = 0; i < m; ++i) {
            det *= a[i*n + i];
        }
        printf("Определитель матрицы = %.3lf\n", det);
    }

    free(a);    // Освобождаем память
    return 0;   // Успешное завершение программы
}

// Приведение вещественной матрицы
// к ступенчатому виду методом Гаусса с выбором
// максимального разрешающего элемента в столбце.
// Функция возвращает ранг матрицы
int gaussMethod(
    int m,          // Число строк матрицы
    int n,          // Число столбцов матрицы
    double *a,      // Адрес массива элементов матрицы
    double eps      // Точность вычислений
) {
    omp_set_num_threads(4);
    int i, j, k, l;
    double r;

    i = 0; j = 0;

    // Распараллеливание
    #pragma omp parallel private (i, j, k, l, r, m, n, a, eps) num_threads(4)
    while (i < m && j < n) {
        // Инвариант: минор матрицы в столбцах 0..j-1
        //   уже приведен к ступенчатому виду, и строка
        //   с индексом i-1 содержит ненулевой эл-т
        //   в столбце с номером, меньшим чем j

        // Ищем максимальный элемент в j-м столбце,
        // начиная с i-й строки
        r = 0.0;
        for (k = i; k < m; ++k) {
            if (fabs(a[k*n + j]) > r) {
                l = k;      // Запомним номер строки
                r = fabs(a[k*n + j]); // и макс. эл-т
            }
        }
        if (r <= eps) {
            // Все элементы j-го столбца по абсолютной
            // величине не превосходят eps.
            // Обнулим столбец, начиная с i-й строки
            for (k = i; k < m; ++k) {
                a[k*n + j] = 0.0;
            }
            ++j;      // Увеличим индекс столбца
            continue; // Переходим к следующей итерации
        }

        if (l != i) {
            // Меняем местами i-ю и l-ю строки
            for (k = j; k < n; ++k) {
                r = a[i*n + k];
                a[i*n + k] = a[l*n + k];
                a[l*n + k] = (-r); // Меняем знак строки
            }
        }

        // Утверждение: fabs(a[i*n + k]) > eps

        // Обнуляем j-й столбец, начиная со строки i+1,
        // применяя элем. преобразования второго рода
        for (k = i+1; k < m; ++k) {
            r = (-a[k*n + j] / a[i*n + j]);

            // К k-й строке прибавляем i-ю, умноженную на r
            a[k*n + j] = 0.0;
            for (l = j+1; l < n; ++l) {
                a[k*n + l] += r * a[i*n + l];
            }
        }

        ++i; ++j;   // Переходим к следующему минору
    }

    return i; // Возвращаем число ненулевых строк
}
