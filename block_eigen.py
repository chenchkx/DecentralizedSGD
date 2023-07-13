import numpy as np

def block_diagonal_matrix(blocks, block_size):
    m, n = blocks.shape[0]*block_size, blocks.shape[1]*block_size
    matrix = np.zeros((m,n))
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = blocks[i,j]
    return matrix

# 创建一个2x2的块矩阵
block_1 = np.array([[1, 1],
                    [1, 1]])

# 使用block_diagonal_matrix函数创建20x20的块对角矩阵
blocks = np.array([[block_1,              np.zeros_like(block_1)],
                   [np.zeros_like(block_1), block_1]])

matrix = block_diagonal_matrix(blocks, block_1.shape[0])
print("块对角矩阵:")
print(matrix)

eigenvalues = np.linalg.eigvals(matrix)
print("特征值:", eigenvalues)

from scipy.linalg import fractional_matrix_power

def normalized_laplacian(adjacency_matrix):
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    normalized_laplacian = fractional_matrix_power(degree_matrix, -0.5) @ laplacian_matrix @ fractional_matrix_power(degree_matrix, -0.5)
    return normalized_laplacian
# 计算归一化拉普拉斯矩阵
laplacian_matrix = normalized_laplacian(matrix)

print("归一化拉普拉斯矩阵:")
print(laplacian_matrix)

laplacian_eigenvalues = np.linalg.eigvals(laplacian_matrix)
print("特征值:", laplacian_eigenvalues)