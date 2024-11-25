
#pca
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[17,7,16,19],
              [12,5,9,21]])
X_transposed = X.T
print("Step 2: Transposed Data:")
print(X_transposed)
X_mean = np.mean(X_transposed, axis=0)
X_centered = X_transposed - X_mean
print("\nStep 3: Centered Data (Subtract Mean):")
print(X_centered)
cov_matrix = np.cov(X_centered, rowvar=False)
print("\nStep 4: Covariance Matrix:")
print(cov_matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nStep 5: Eigenvalues:")
print(eigenvalues)
print("\nStep 5: Eigenvectors:")
print(eigenvectors)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
print("\nStep 6: Sorted Eigenvalues:")
print(sorted_eigenvalues)
print("\nStep 6: Sorted Eigenvectors:")
print(sorted_eigenvectors)
X_pca_manual = np.dot(X_centered, sorted_eigenvectors)
print("\nStep 7: Projected Data (Manual PCA):")
print(X_pca_manual)
plt.figure(figsize=(8, 6))
plt.scatter(X_transposed[:, 0], X_transposed[:, 1], color='blue', label='Original Data')
plt.scatter(X_centered[:, 0], X_centered[:, 1], color='red', label='Centered Data')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA Transformation with X and Y Axes')
plt.legend()
plt.grid(True)
plt.show()

