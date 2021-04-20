import random
from typing import List


def generate_adjacency_matrix(size: int, max: int = 100) -> List[List[int]]:
    """
    Generate adjacency matrix of dimension (size, size)

    Parameters:
        size (int): dimension of adjacency matrix
        max (int): max value of int in the matrix

    Returns:
        list[list[int]]: adjacency matrix
    """
    res = [[0 for i in range(size)] for j in range(size)]

    for i in range(size):
        for j in range(i):
            res[i][j] = random.randint(0, max)
            res[j][i] = res[i][j]

    return res
