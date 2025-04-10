import numpy as np

if __name__ == '__main__':
    identities = [np.eye(n) for n in range(1, 6)]
    for identity in identities:
        print(f"Identity matrix of size {identity.shape[0]}:")
        eigenvalues, eigenvectors = np.linalg.eig(identity)
        print(f"Eigenvalues: {eigenvalues}")
        for i, eigenvector in enumerate(eigenvectors):
            print(f"Eigenvector {i}: {eigenvector}")
        print()