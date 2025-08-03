# Regression Models Comparison

This project provides a **comprehensive benchmark** of regression algorithms—implemented in both **analytical closed-form solutions** and **numerical gradient descent optimization**—on the **House Prices: Advanced Regression Techniques** dataset. It includes robust preprocessing, categorical encoding strategies, model training, and detailed performance evaluations.



## 1. Overview

This repository demonstrates:

* End-to-end pipeline for handling mixed-type housing data
* Comparison of **four regression algorithms**:

  1. **Linear Regression**
  2. **Ridge Regression** (L2 regularization)
  3. **Lasso Regression** (L1 regularization)
  4. **Elastic Net** (combined L1 and L2 regularization)
* **Dual implementations**:

  * **Analytical (Closed-form)**: Using Scikit-learn
  * **Numerical (Gradient Descent)**: Using TensorFlow/Keras
* Evaluation via statistical metrics and learned model parameters



## 2. Project Structure

```
project_root/
│── data/
│   └── test.csv
│   └── train.csv
│── notebooks/
│   └── 01_eda.ipynb
│   └── 02_regression.ipynb
│── requirements.txt             
│── READMEd.m                   # Project documentation
```



## 3. Dataset

* **Source**: [House Prices: Advanced Regression Techniques - Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
* **Target variable**: `SalePrice`
* **Feature types**:

  * **Nominal categorical**: Zone, Street type, Neighborhood, etc.
  * **Ordinal categorical**: Quality ratings, Condition scales
  * **Numerical**: Lot size, living area, garage size, etc.



## 4. Data Preprocessing

### 4.1 Feature Classification

* **Nominal**: Non-ordered categories → One-hot encoded
* **Ordinal**: Ordered categories → Integer-mapped
* **Continuous/Discrete numeric**: Used directly

### 4.2 Encoding

* **One-hot encoding** for nominal categorical variables
* **Ordinal mapping** with semantic integer scoring

### 4.3 Scaling

* Standardization with `StandardScaler` to ensure stable training



## 5. Regression Models

Each regression algorithm is implemented in **two approaches**:

### 5.1 Linear Regression

* **Analytical**: `LinearRegression`
* **Numerical**: Keras dense layer, optimized with Adam + MSE

### 5.2 Ridge Regression

* **Analytical**: `Ridge` with L2 penalty
* **Numerical**: Keras dense layer with `l2` regularizer

### 5.3 Lasso Regression

* **Analytical**: `Lasso` with L1 penalty
* **Numerical**: Keras dense layer with `l1` regularizer

### 5.4 Elastic Net

* **Analytical**: `ElasticNet` with combined L1 and L2
* **Numerical**: Keras dense layer with `l1_l2` regularizer



## 6. Training Configuration

* **Data split**: 80% training / 20% validation
* **Optimizer**: Adam (`lr=0.001`)
* **Batch size**: 512
* **Epochs**: 10,000
* **Loss**: Mean Squared Error (MSE)
* **Metric**: Mean Absolute Error (MAE)



## 7. Evaluation Metrics

* **R² Score**: Proportion of variance explained
* **MAE**: Average prediction error
* **Coefficients & Intercept**: Feature weight analysis
* Side-by-side comparison between analytical and numerical solutions



## 8. Results & Interpretation

* **Ridge Regression**: Reduces variance and mitigates overfitting
* **Lasso Regression**: Enforces sparsity, enabling feature selection
* **Elastic Net**: Combines benefits of Ridge and Lasso -> High performane 
* **Analytical vs Numerical**:

  * Analytical: Faster on small to moderate datasets
  * Numerical: Scalable for high-dimensional, large datasets

### Visualization (Optional)

Plots can be generated to visualize:

* R² scores comparison across models
* Coefficient shrinkage effects in Lasso and Elastic Net
* Loss curves during numerical optimization



## 9. Installation

```bash
git clone https://github.com/honggquan24/regression_comparison
cd regression_comparison
pip install -r requirements.txt
```

Requirements:

* Python 3.8+
* NumPy, Pandas, Scikit-learn, TensorFlow, Matplotlib



## 10. Usage

### Use Jupyter Notebook

```bash
jupyter notebook notebooks/regression_comparison.ipynb
```