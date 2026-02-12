# Import statement
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


## Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print(df.head(5))

#Data Info
df.info()
df.shape

######### Dataset characteristics: #########
#  - Total samples: 569
#  - Number of features: 30 (all numeric)
#  - Target classes:
#        0 -> malignant
#        1 -> benign

# In this investigation, benign (label = 1) is treated as the positive class.

print(df.shape)
print(data.target_names)
df.describe()
print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(df, y,train_size=0.8,random_state=42, stratify=y)

# The dataset is divided into:
# - 80% Training data
# - 20% Testing data

# Stratified sampling is used to preserve the proportion of malignant and benign cases in both sets. The testing set is used only for final evaluation.

selector = SelectKBest(score_func=f_classif, k=5)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

print("Selected feature indices:", selector.get_support(indices=True))

# To reduce dimensionality and prevent overfitting, we select the top 5 most significant features using univariate feature selection.
# Benefits:
# - Reduces model complexity
# - Improves generalization
# - Reduces computational cost

# Only the training data is used to fit the selector to avoid data leakage.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

# SVM is sensitive to feature scale because it relies on distance calculations.
# Therefore, features are standardized to have:
# - Mean = 0
# - Standard deviation = 1

# The scaler is fitted only on training data and then applied to both training and testing sets.

############### SVM MODELS ###############

# We evaluate multiple SVM kernels:

# 1. Linear Kernel
# 2. Polynomial Kernel (degree 2, 3, 4)
# 3. Gaussian (RBF) Kernel with different gamma values
# 4. Sigmoid Kernel

# We also investigate the effect of regularization parameter C for the linear kernel.

# Each kernel produces a different decision boundary:
# - Linear: Straight hyperplane
# - Polynomial: Curved boundary
# - RBF: Highly flexible nonlinear boundary
# - Sigmoid: Similar to neural network activation

kernels = {
    "Linear (C = 0.1)": SVC(kernel="linear", random_state=42, C = 0.1), ## Linear Kernel C = 0.1
    "Linear (C = 1)": SVC(kernel="linear", random_state=42, C = 1), ## Linear Kernel C = 1
    "Linear (C = 10)": SVC(kernel="linear", random_state=42, C = 10), ## Linear Kernel C = 10
    "Poly (degree=2)": SVC(kernel="poly", degree=2, random_state=42), ## Polynomial Kernel degree = 2
    "Poly (degree=3)": SVC(kernel="poly", degree=3, random_state=42), ## Polynomial Kernel degree = 3
    "Poly (degree=4)": SVC(kernel="poly", degree=4, random_state=42), ## Polynomial Kernel degree = 4
    "Gaussian (RBF,Gamma - 0.01)": SVC(kernel="rbf", gamma=0.01, random_state=42), ## Gaussian Kernel Gamma = 0.01
    "Gaussian (RBF),Gamma - 0.1": SVC(kernel="rbf", gamma=0.1, random_state=42), ## Gaussian Kernel Gamma = 0.1
    "Gaussian (RBF),Gamma - 1.0": SVC(kernel="rbf", gamma=1, random_state=42), ## Gaussian Kernel Gamma = 1
    "Sigmoid_g0.01_c0": SVC(kernel="sigmoid", random_state=42, gamma=0.01, coef0=0), ## Sigmoid Kernel Gamma = 0.01 Coef = 0
    "Sigmoid_g0.1_c1": SVC(kernel="sigmoid", random_state=42, gamma=0.1, coef0=1) ## Sigmoid Kernel Gamma = 0.1 Coef = 1
}

results = []

for name, model in kernels.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Manual metric calculations
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results.append({
        "Kernel": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"\nKernel: {name}")
    print("Confusion Matrix:\n", cm)
    print(f"TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


results_df = pd.DataFrame(results)
print(results_df)

# The SVM kernel comparison shows that Linear (C = 10) and RBF (Gamma = 1.0) achieved the best overall performance, with the highest accuracy (96.49%) and F1 score (0.9722), indicating an excellent balance between precision and recall. Increasing the regularization parameter C improved linear kernel performance, suggesting the dataset is close to linearly separable when properly tuned. 

# The rest of the RBF kernels also performed very strongly, especially with higher gamma values, demonstrating its ability to model nonlinear decision boundaries effectively. Polynomial kernels of degree 2 and 4 showed lower accuracy and precision despite achieving perfect recall, indicating overfitting and excessive false positives, while the sigmoid kernel produced stable but slightly inferior results compared to the top models. 

# Overall, the results suggest that either a well-regularized linear SVM or an appropriately tuned RBF kernel is most suitable for this classification problem.

######### Plots ########

## F1 Score

plt.figure(figsize=(10, 5))
plt.bar(results_df["Kernel"], results_df["F1 Score"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1 Score")
plt.title("SVM Kernel Comparison (F1 Score)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("F1Score.png")
plt.show()

## Complete performance

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(
        results_df["Kernel"],
        results_df[metric],
        marker="o",
        label=metric
    )

plt.xticks(rotation=45, ha="right")
plt.ylabel("Score")
plt.title("SVM Kernel Performance Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("CompletePerformance.png")
plt.show()
