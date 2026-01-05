# Diabetes Dataset
# 🏥 Diabetes Prediction & Risk Analysis using Machine Learning

## 📌 Project Overview
This project applies supervised and unsupervised Machine Learning techniques to predict the risk of Diabetes in a dataset of **200,000 patients**. 

The goal was to compare how different algorithms handle **highly imbalanced medical data** (where healthy patients vastly outnumber sick ones) and to explore natural patient groupings using clustering.

## 📂 Dataset
- **Size:** 200,000 entries
- **Features:** 12 clinical features including Age, BMI, Cholesterol, Blood Pressure, Heart Rate, and Smoking status.
- **Target:** `Diabetes` (Binary Classification: Yes/No)
- **Challenge:** The dataset is heavily imbalanced, requiring careful metric evaluation beyond standard accuracy.

## ⚙️ Methodology
### 1. Data Preprocessing
- **Imputation:** Handled missing numerical values using Mean Imputation.
- **Encoding:** Applied Label Encoding for binary targets and One-Hot Encoding for categorical features (Blood Type).
- **Scaling:** Applied `StandardScaler` to normalize features like Cholesterol and Age for Neural Network performance.

### 2. Models Implemented
| Model | Type | Purpose |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear | Baseline model to establish linear decision boundaries. |
| **Decision Tree** | Non-Linear | Interpretable model to capture specific "If-Then" risk rules. |
| **Neural Network (MLP)** | Deep Learning | Multi-layer Perceptron to capture complex, non-linear feature interactions. |
| **K-Means Clustering** | Unsupervised | Grouped patients into clusters to find hidden patterns without labels. |

## 📊 Results & Evaluation
We evaluated models using **F1-Score** and **AUC-ROC** rather than just Accuracy, as accuracy is misleading for imbalanced data.

- **Logistic Regression:** Best balance of sensitivity (Recall) and precision.
- **Decision Tree:** High accuracy but prone to overfitting without pruning; excellent interpretability.
- **Neural Network:** Achieved high accuracy (90%+) but struggled with the minority class (Diabetes cases), highlighting the difficulty of "Black Box" models on imbalanced data without extensive tuning.
- **Clustering:** K-Means successfully identified distinct patient subgroups based on metabolic markers.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the Jupyter Notebook `Diabetes_Analysis.ipynb`.
