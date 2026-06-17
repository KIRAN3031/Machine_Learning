# Supervised Machine Learning

This section contains implementations of supervised learning algorithms where the model learns from labeled training data.

## 📂 Modules

### 1. Linear Regression
Learn how to predict continuous values using linear relationships.

**Contents:**
- `Single_Linear_Regression/` - Simple linear regression with one feature
- `Mutliple_Linear_Regression/` - Multiple linear regression with multiple features

**What you'll learn:**
- Fitting linear models to data
- Understanding cost functions
- Gradient descent optimization
- Evaluating regression models
- Real estate price prediction, sales forecasting

### 2. Classification
Classify data into discrete categories (binary or multi-class).

**Contents:**
- `logistic_regression/` - Logistic regression for binary/multi-class classification
- `app/` - Practical applications

**What you'll learn:**
- Logistic function and sigmoid curve
- Binary and multi-class classification
- Confusion matrix and classification metrics
- ROC curves and AUC
- Threshold optimization

### 3. Decision Tree
Tree-based models that make decisions through sequential splits.

**What you'll learn:**
- Information gain and entropy
- Gini index
- Tree construction and pruning
- Handling categorical and numerical data
- Interpretable model decisions

### 4. K-Nearest Neighbors (KNN)
Instance-based learning using similarity to nearest neighbors.

**What you'll learn:**
- Distance metrics (Euclidean, Manhattan, etc.)
- Finding optimal k value
- Handling imbalanced data
- Computational efficiency
- Applications in classification and regression

### 5. Naive Bayes Classifier
Probabilistic classifier based on Bayes' theorem.

**What you'll learn:**
- Conditional probability
- Independence assumption
- Gaussian, Multinomial, and Bernoulli variants
- Text classification
- Spam detection

### 6. Random Forest
Ensemble method combining multiple decision trees.

**What you'll learn:**
- Bootstrap aggregating (Bagging)
- Random feature selection
- Combining predictions
- Feature importance
- Advantages over single trees

### 7. Support Vector Machine (SVM)
Powerful algorithm for classification and regression.

**What you'll learn:**
- Margin maximization
- Support vectors
- Kernel trick (Linear, RBF, Polynomial)
- One-vs-Rest strategy for multi-class
- Soft margins and C parameter

### 8. Ensemble Learning
Techniques to combine multiple models for better performance.

**What you'll learn:**
- Bagging
- Boosting (AdaBoost, Gradient Boosting)
- Voting classifiers
- Stacking
- Model diversity

### 9. Regularization
Techniques to prevent overfitting and improve generalization.

**What you'll learn:**
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- Elastic Net
- Early stopping
- Dropout

## 🎯 Learning Path

1. **Start here:** Linear_Regression - Build fundamentals
2. **Then:** classification - Learn decision boundaries
3. **Next:** Decision_tree, KNN - Explore different approaches
4. **Advanced:** Random_forest, SVM - Master powerful algorithms
5. **Optimization:** Regularization, Ensemble_learning - Improve performance

## 💻 Requirements

```python
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

## 📊 Common Algorithms Comparison

| Algorithm | Type | Best For | Complexity |
|-----------|------|----------|-----------|
| Linear Regression | Regression | Linear relationships | O(n) |
| Logistic Regression | Classification | Binary/Multi-class | O(n) |
| Decision Tree | Both | Non-linear, interpretable | O(n log n) |
| KNN | Both | Local patterns | O(n) |
| SVM | Both | High-dimensional data | O(n²) to O(n³) |
| Random Forest | Both | General purpose | O(n log n) |
| Naive Bayes | Classification | Fast classification | O(n) |

## 🔍 Evaluation Metrics

### For Regression
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### For Classification
- Accuracy
- Precision & Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## 🚀 Quick Start

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [Fast.ai](https://www.fast.ai/)
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/)

## ✨ Tips

- Always scale/normalize your features
- Use cross-validation for robust evaluation
- Monitor for overfitting and underfitting
- Feature engineering is crucial
- Start simple, then add complexity
- Understand your data before modeling

---

**Happy Learning!** 🎓
