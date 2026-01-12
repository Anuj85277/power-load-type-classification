This project focuses on building a machine learning classification model to predict the Load Type of a power system based on historical energy consumption and power-related features.
The load types are categorized into:

Light_Load

Medium_Load

Maximum_Load

The objective is to accurately classify the load condition using real-world power system data while following a time-based validation strategy.

Objective

To develop a machine learning model that predicts the Load_Type of a power system using historical data, applying proper data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

📂 Dataset Description

The dataset contains time-series power system measurements with the following features:

Feature	Description
Date_Time	Timestamp of the measurement
Usage_kWh	Industry energy consumption (kWh)
Lagging_Current_Reactive.Power_kVarh	Lagging reactive power
Leading_Current_Reactive_Power_kVarh	Leading reactive power
CO2(tCO2)	CO₂ emissions
Lagging_Current_Power_Factor	Lagging power factor
Leading_Current_Power_Factor	Leading power factor
NSM	Number of seconds from midnight
Load_Type	Target variable (Light / Medium / Maximum)
🧠 Approach & Methodology
1️⃣ Data Preprocessing

Converted Date_Time into datetime format

Extracted Month and Hour features

Handled missing values using median imputation

Encoded categorical target variable (Load_Type)

2️⃣ Exploratory Data Analysis (EDA)

Analyzed load type distribution

Visualized feature correlations using a heatmap

3️⃣ Feature Engineering

Created time-based features (Month, Hour)

Scaled numerical features using StandardScaler

4️⃣ Model Selection

Used Random Forest Classifier, well-suited for tabular data

Applied class balancing to handle any class imbalance

5️⃣ Validation Strategy

Time-based split

The last month of data was used as the test set to simulate real-world prediction on unseen, recent data

📊 Model Performance
✅ Final Model: Random Forest Classifier

Accuracy: 93.55%

Achieved balanced precision, recall, and F1-score across all load classes

Demonstrated strong generalization on unseen data

📈 Evaluation Metrics Used

Accuracy

Precision

Recall

F1-score

Confusion Matrix

🏆 Final Conclusion

The Random Forest model successfully classified power system load types with high accuracy using a time-based validation approach. The model demonstrated robust performance and reliability, making it suitable for real-world load prediction scenarios.

🛠️ Tech Stack

Python
 
Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

VS Code

▶️ How to Run the Project

Clone the repository:

git clone <repo-link>


Create and activate virtual environment:

python -m venv ml_env
ml_env\Scripts\activate


Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

