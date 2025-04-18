import numpy as np
import matplotlib.pyplot as plt

def tensor_product(arrays):
    """
    Computes the tensor product of a list of arrays
    """
    result = arrays[0]
    for array in arrays[1:]:
        result = np.kron(result, array)
    return result

def pauli_spin_matrices():
    """
    Returns the Pauli spin matrices X,Y and Z
    """
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return X, Y, Z

def creation_operator(j:int, N:int) -> np.ndarray:
    """
    Returns the creation operator for a particle at site j for a N-site system
    """
    X,Y,Z = pauli_spin_matrices()
    I = np.eye(2)
    operators = [I] * N

    z_op = operators.copy()
    for i in range(j):
        z_op[i] = Z
    
    z_op = tensor_product(z_op)
    x_op = operators.copy()
    x_op[j] = X
    x_op = tensor_product(x_op)
    y_op = operators.copy()
    y_op[j] = Y
    y_op = tensor_product(y_op)
    return 0.5 * z_op @ (x_op + 1j*y_op)

def annihilation_operator(j:int, N:int) -> np.ndarray:
    """
    Returns the annihilation operator for a particle at site j for a N-site system
    """
    X,Y,Z = pauli_spin_matrices()
    I = np.eye(2)
    operators = [I] * N

    z_op = operators.copy()
    for i in range(j):
        z_op[i] = Z
    
    z_op = tensor_product(z_op)
    x_op = operators.copy()
    x_op[j] = X
    x_op = tensor_product(x_op)
    y_op = operators.copy()
    y_op[j] = Y
    y_op = tensor_product(y_op)
    return 0.5 * z_op @ (x_op - 1j*y_op)

def majorana_ops(j:int, N:int) -> np.ndarray:
    """
    Returns the Majorana operators for a particle at site j for a N-site system
    Order (gamma_{2j-1}, gamma_{2j})
    """
    cj = annihilation_operator(j, N)
    c_dagger_j = creation_operator(j, N)
    gamma1 = cj + c_dagger_j
    gamma2 = 1j * (cj - c_dagger_j)
    return gamma1, gamma2

def exchange_op(k: int, l: int, N: int) -> np.ndarray:
    """
    Returns the braid operator that exchanges Majoranas at positions k and l
    (indices refer to Majorana numbers: from 0 to 2N-1)
    """
    if k == l:
        raise ValueError("k and l must be different")

    majoranas = []
    for j in range(N):
        gamma1, gamma2 = majorana_ops(j, N)
        majoranas.append(gamma1)
        majoranas.append(gamma2)

    gamma_k = majoranas[k]
    gamma_l = majoranas[l]

    # Braid operator: up to a global phase, valid for Majoranas that anticommute
    B_kl = (np.eye(2**N) + gamma_k @ gamma_l) / np.sqrt(2)
    return B_kl
























# def test_majorana_algebra(N: int, tol: float = 1e-10):
#     """
#     Tests whether the Majorana operators satisfy the canonical anticommutation relations:
#     {gamma_i, gamma_j} = 2 * delta_ij * I
#     """
#     print(f"Testing Majorana algebra for N = {N}...")

#     dim = 2**N
#     all_gammas = []
#     for j in range(N):
#         gamma1, gamma2 = majorana_ops(j, N)
#         all_gammas.append(gamma1)
#         all_gammas.append(gamma2)

#     success = True
#     for i in range(2*N):
#         for j in range(2*N):
#             anti_comm = all_gammas[i] @ all_gammas[j] + all_gammas[j] @ all_gammas[i]
#             expected = 2 * np.eye(dim) if i == j else np.zeros((dim, dim))
#             if not np.allclose(anti_comm, expected, atol=tol):
#                 print(f"❌ Failed at (i, j) = ({i}, {j})")
#                 print("Computed:\n", anti_comm)
#                 print("Expected:\n", expected)
#                 success = False

#     if success:
#         print("✅ All Majorana anticommutation relations passed!")

# def test_braiding_double_actions(N=2, tol=1e-10):
#     """
#     Tests whether double exchanges act like Pauli matrices on the logical qubit space:
#     B_12^2 ~ σ^z
#     B_23^2 ~ σ^x
#     """
#     print(f"Testing B_ij^2 actions on logical qubit for N={N}...")

#     dim = 2**N
#     B12 = exchange_op(0, 1, N)
#     B23 = exchange_op(1, 2, N)
#     B34 = exchange_op(2, 3, N)
#     B12_sq = B12 @ B12
#     B23_sq = B23 @ B23

#     # Logical qubit basis: even parity states |00>, |11>
#     basis = [np.zeros((dim,), dtype=complex) for _ in range(2)]
#     basis[0][0] = 1.0  # |00> in computational basis
#     basis[1][-1] = 1.0 # |11> is last basis state in even parity

#     B12_proj = np.array([[np.vdot(bi, B12_sq @ bj) for bj in basis] for bi in basis])
#     B23_proj = np.array([[np.vdot(bi, B23_sq @ bj) for bj in basis] for bi in basis])

#     # Pauli matrices
#     sz = np.array([[1, 0], [0, -1]])
#     sx = np.array([[0, 1], [1, 0]])

#     def compare_up_to_phase(A, B, name):
#         phase = np.trace(np.dot(np.linalg.inv(B), A)) / 2  # crude phase matching
#         if np.allclose(A, phase * B, atol=tol):
#             print(f"✅ {name} matches expected Pauli up to global phase.")
#         else:
#             print(f"❌ {name} does NOT match expected Pauli.")
#             print("Got:\n", A)
#             print("Expected:\n", phase * B)

#     compare_up_to_phase(B12_proj, sz, "B_12^2")
#     compare_up_to_phase(B23_proj, sx, "B_23^2")



# test_majorana_algebra(3)
# test_braiding_double_actions(N=3)
