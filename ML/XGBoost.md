# 🧠 XGBoost: Overview, Algorithm, and How to Use It

## 💡 What is XGBoost?

**XGBoost (Extreme Gradient Boosting)** is a high-performance, scalable implementation of gradient-boosted decision trees. It is widely used for:

- 📈 Alpha signal prediction  
- ⚠️ Risk modeling  
- 💳 Credit risk / fraud detection  
- 🧠 Execution modeling  

It builds an **ensemble of shallow trees**, each trained to correct the **residual errors** of the previous model — this is the essence of **boosting**.

---

## ⚙️ Core Principle: Additive Tree Boosting

At each iteration \( t \), we add a new tree \( f_t \) to minimize the total loss:

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)
$$

Where:

- \( \hat{y}_i^{(t)} \): prediction at iteration \( t \)  
- \( f_t(x_i) \): the new tree  
- Each new tree learns to correct the residuals (gradients)

---

## 🔥 Loss Functions in XGBoost

| Task                  | Loss Function                         | Example                                        |
|-----------------------|----------------------------------------|------------------------------------------------|
| **Regression**        | Mean Squared Error (default)           | $$ (y - \hat{y})^2 $$                          |
| **Classification**    | Logistic / Log Loss                    | $$ -[y\log(p) + (1-y)\log(1-p)] $$             |
| **Ranking**           | Pairwise loss (LambdaRank, etc.)       | Used in recommendation or stock ranking        |
| **Custom Objective**  | Define your own                        | e.g. asymmetric loss for trading               |

---

## 🧠 Regularization in XGBoost

The objective function is:

$$
\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^t \Omega(f_k)
$$

Where:

- \( l \): loss function (MSE, logistic, etc.)  
- $$ \Omega(f_k) = \gamma T + \frac{1}{2} \lambda \sum_j w_j^2 $$

**Terms:**

- \( \gamma \): Penalty for the number of leaves — controls tree complexity  
- \( \lambda \): L2 regularization on leaf weights — helps prevent overfitting

---

## 🧮 Taylor Approximation (2nd Order)

XGBoost uses a **second-order Taylor expansion** for fast optimization:

$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)
$$

Where:

- $$ g_i = \frac{\partial l}{\partial \hat{y}_i} $$ (gradient)  
- $$ h_i = \frac{\partial^2 l}{\partial \hat{y}_i^2} $$ (hessian)

---

## 🌳 Tree Building and Split Gain

Split Gain formula for a tree node:

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

- \( G, H \): sums of gradients and hessians  
- Split is chosen if gain is large enough

---

## 🛠️ Example: How to Use XGBoost in Python (Quant Style)

```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Your factor data (features) and target (label)
X = factor_matrix  # e.g., standardized momentum, value, etc.
y = future_return  # label, such as next 5-day return

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize model
model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0, # L2 regularization
    objective="reg:squarederror"
)

# Train model
model.fit(X_train, y_train)

# Predict future return
predicted = model.predict(X_test)
