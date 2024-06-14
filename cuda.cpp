#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

__global__ void gaussianEliminationCUDA(double *matrix, int numRows, int numCols)
{
    const int pivot = blockIdx.x * blockDim.x + threadIdx.x;
    if (pivot < numCols)
    {
        double maxVal = fabs(matrix[pivot * (numCols + 1) + pivot]);
        int maxRow = pivot;
        for (int row = pivot + 1; row < numRows; ++row)
        {
            double absVal = fabs(matrix[row * (numCols + 1) + pivot]);
            if (absVal > maxVal)
            {
                maxVal = absVal;
                maxRow = row;
            }
        }
        if (maxRow != pivot)
        {
            for (int col = pivot; col <= numCols; ++col)
            {
                double temp = matrix[pivot * (numCols + 1) + col];
                matrix[pivot * (numCols + 1) + col] = matrix[maxRow * (numCols + 1) + col];
                matrix[maxRow * (numCols + 1) + col] = temp;
            }
        }
        __syncthreads();
        for (int row = pivot + 1; row < numRows; ++row)
        {
            double factor = matrix[row * (numCols + 1) + pivot] / matrix[pivot *
                                                                             (numCols + 1) +
                                                                         pivot];
            for (int col = pivot; col <= numCols; ++col)
            {
                matrix[row * (numCols + 1) + col] -= factor * matrix[pivot *
                                                                         (numCols + 1) +
                                                                     col];
            }
        }
        __syncthreads();
    }
    if (pivot < numCols)
    {
        matrix[pivot * (numCols + 1) + numCols] /= matrix[pivot * (numCols + 1) +
                                                          pivot];
        matrix[pivot * (numCols + 1) + pivot] = 1.0;
        for (int row = pivot - 1; row >= 0; --row)
        {
            double factor = matrix[row * (numCols + 1) + pivot];
            matrix[row * (numCols + 1) + pivot] = 0.0;
            matrix[row * (numCols + 1) + numCols] -= factor * matrix[pivot *
                                                                         (numCols + 1) +
                                                                     numCols];
        }
    }
}

void gaussianElimination(std::vector<std::vector<double>> &matrix)
{
    const int numRows = matrix.size();
    const int numCols = matrix[0].size() - 1;
    double *matrixData = new double[numRows * (numCols + 1)];
    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col <= numCols; ++col)
        {
            matrixData[row * (numCols + 1) + col] = matrix[row][col];
        }
    }
    double *deviceMatrixData;
    cudaMalloc((void **)&deviceMatrixData, numRows * (numCols + 1) *
                                               sizeof(double));
    cudaMemcpy(deviceMatrixData, matrixData, numRows * (numCols + 1) * sizeof(double), cudaMemcpyHostToDevice);
    const int numThreads = 256;
    const int numBlocks = (numCols + numThreads - 1) / numThreads;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    gaussianEliminationCUDA<<<numBlocks, numThreads>>>(deviceMatrixData, numRows,
                                                       numCols);
    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec -
                                                        start.tv_usec) /
                                                           1000000.0;
    std::cout << "Elapsed Time: " << elapsedTime << " seconds" << std::endl;
    cudaMemcpy(matrixData, deviceMatrixData, numRows * (numCols + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col <= numCols; ++col)
        {
            matrix[row][col] = matrixData[row * (numCols + 1) + col];
        }
    }
    delete[] matrixData;
    cudaFree(deviceMatrixData);
}

int main(int agrc, char *argv[])
{
    int n;
    double eps, buff;
    std::vector<std::vector<double>> matrix;

    printf("Введите размеры матрицы n: ");
    scanf("%d", &n);
    printf("\n");

    printf("Введите точность вычислений eps: ");
    scanf("%lf", &eps);
    printf("\n");

    printf("Введите элементы матрицы: ");
    for (size_t i = 0; i < n; i++)
    {
        matrix.push_back(std::vector<double>());
        for (size_t j = 0; j < n; j++)
        {
            scanf("%lf", &buff);
            matrix[i].push_back(buff);
        }
    }

    printf("==================================\n");
    printf("Тест функции...\n");

    // инициализируем события
    cudaEvent_t start, stop;
    float elapsedTime;

    // создаем события
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // запись события старта
    cudaEventRecord(start, 0);

    gaussianElimination(matrix);

    // запись события окончания работы
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // вывод информации
    printf("Метод Гаусса - Cuda. Время выполнения: %5.5f s\n", elapsedTime);
    printf("==================================\n");
    // уничтожение события
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	return(0);
}