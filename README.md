# Obesity Prediction Web App: Machine Learning for Health Insights

This project develops an interactive web application that predicts obesity categories based on individual health metrics. Leveraging machine learning models, it provides users with personalized insights into their fitness levels.

## Project Goal & What It Does

The primary goal is to build a user-friendly tool for health awareness by predicting an individual's obesity category. The web application allows users to input their data and receive an instant prediction, showcasing the practical application of machine learning in public health and fitness monitoring.

Key functionalities include:
* **Interactive Prediction:** Users can input their Age, Gender, Height, Weight, BMI, and Physical Activity Level to get a real-time obesity prediction.
* **Machine Learning Models:** Utilizes robust **Random Forest Classifier** and **Logistic Regression** models for classification.
* **Data Analysis & Visualization:** Includes comprehensive data exploration, visualization of distributions, and correlation analysis.
* **Model Evaluation:** Provides accuracy metrics and confusion matrices to assess model performance.

## Technical Highlights

* **Dataset:** Uses `obesity_data.csv`, a dataset containing various health and physical activity metrics (Age, Gender, Height, Weight, BMI, PhysicalActivityLevel) alongside an `ObesityCategory` target variable.
* **Data Preprocessing:**
    * **Categorical Encoding:** `Gender` column is transformed using `LabelEncoder`.
    * **Feature Scaling:** Numerical features are scaled using `StandardScaler` to optimize model performance.
* **Model Training:**
    * **Baseline:** Demonstrates model training with `LogisticRegression`.
    * **Main Model:** Employs `RandomForestClassifier` for its robust performance.
    * Models are trained on a split dataset (`train_test_split`) for rigorous evaluation.
* **Model Persistence:** Trained machine learning models are saved using `pickle` for later loading and deployment within the web application, avoiding retraining.
* **Web Application Framework:** Deployed as an interactive web application using **Streamlit**, providing a user-friendly interface for input and prediction display.
* **Evaluation Metrics:** Model performance is rigorously evaluated using `accuracy_score`, `confusion_matrix`, and `classification_report`. Visualizations (e.g., accuracy bar plot, confusion matrix heatmap) highlight performance.

## How to Run This Project

To run this project and interact with the Obesity Prediction Web App:

1.  **Upload the Dataset:** Ensure `obesity_data.csv` is uploaded to your Google Colab environment or the project directory.
2.  **Upload Trained Models:** Ensure `RandomForestClassifier.pkl` and `LogisticRegression.pkl` (generated from training the models in the notebook) are available in your Colab environment or the project directory.
3.  **Open in Google Colab:**
    * Go to `colab.research.google.com`.
    * Navigate to `File > Open notebook > GitHub`, and paste your repository URL (if hosted on GitHub) or upload the notebook directly.
4.  **Install Libraries:**
    * Run the `pip install streamlit scikit-learn` command within a Colab cell.
5.  **Run the Streamlit App:**
    * Ensure your Streamlit app script (e.g., `paginaDePredict.py` as inferred from the code) is present in your Colab environment.
    * Execute the commands to run the Streamlit app:
        ```bash
        ! wget -q -O - ipv4.icanhazip.com
        ! streamlit run paginaDePredict.py & npx localtunnel --port 8501
        ```
    * Click on the provided `your url is: https://[...].loca.lt` link in the output to access the web application in your browser.

## Evaluation Results (from notebook output)

* **RandomForestClassifier accuracy:** 1.000
* **LogisticRegression accuracy:** 0.956

*(Note: Screenshots of plots like the accuracy bar plot and RandomForest Confusion Matrix would be included here in a typical GitHub README.)*

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn (`sklearn`)
* Streamlit
* Google Colaboratory
* `pickle`
