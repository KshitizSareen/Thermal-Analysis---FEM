import numpy as np

def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Fast, CPU matrix multiplication using NumPy/BLAS.
    Shapes: (m, k) @ (k, n) -> (m, n)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Inner dimensions must match: {A.shape} @ {B.shape}")
    return A @ B 

def matrix_mult_gpu(A: np.ndarray,B: np.ndarray) -> np.ndarray:
    


if __name__ == "__main__":
    m, k, n = 256, 384, 128
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)

    C1 = matmul_numpy(A, B)