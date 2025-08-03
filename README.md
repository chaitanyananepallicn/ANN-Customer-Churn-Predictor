# ANN-Customer-Churn-Predictor

# ANN Customer Churn Predictor

This project uses an Artificial Neural Network (ANN) to predict customer churn based on various customer attributes. A web application built with Streamlit allows for real-time predictions.

## Live Application

**You can access the live deployed application here:**

[**➡️ Live Churn Predictor App**](https://ann-customer-churn-predictor-o5bujsabjsz29u5dtwuppe.streamlit.app/)

---

## Dataset

The project utilizes the "Churn_Modelling.csv" dataset. The dataset contains 10,000 records and 14 columns, including customer ID, credit score, geography, gender, age, tenure, balance, number of products, credit card status, active member status, estimated salary, and whether the customer has exited. Unnecessary columns like 'RowNumber', 'CustomerId', and 'Surname' are dropped during preprocessing.

## Model and Preprocessing

The model is a Sequential Artificial Neural Network built with TensorFlow and Keras.

* **Preprocessing**:
    * The 'Gender' feature is converted into numerical format using `LabelEncoder`.
    * The categorical 'Geography' feature is transformed into numerical data using `OneHotEncoder`.
    * All features are scaled using `StandardScaler` before being fed into the model.

* **Architecture**:
    * The network consists of an input layer, two hidden layers with 64 and 32 neurons respectively, and an output layer.
    * The 'relu' activation function is used in the hidden layers, while the 'sigmoid' function is used for the final output layer to produce a probability score.
    * The model is compiled using the 'adam' optimizer and 'binary_crossentropy' as the loss function.

## Web Application

A user-friendly web application has been developed using Streamlit to interact with the trained model. Users can input customer details such as geography, gender, age, balance, credit score, and other relevant information through a simple UI. The application then processes this data, feeds it to the model, and displays the churn probability, indicating whether the customer is likely to churn.

## How to Run the Project Locally

Follow these steps in your terminal. The order is important.

1.  **Clone the repository**:
    *This downloads the project files to your computer.*
    ```bash
    git clone [https://github.com/chaitanyananepallicn/ANN-Customer-Churn-Predictor.git](https://github.com/chaitanyananepallicn/ANN-Customer-Churn-Predictor.git)
    ```
    
2.  **Install the required dependencies**:
    *This command reads the `requirements.txt` file and installs all the necessary Python libraries.*
    ```bash
    pip install -r requirements.txt
    ```

3. **Navigate into the project directory**:
    *You must be inside this folder to access the necessary files.*
    ```bash
    cd ANN-Customer-Churn-Predictor
    ```
    
4.  **Run the Streamlit application**:
    ```bash
    streamlit run ChurnModeling-ANN/app.py
    ```
5.  Open your web browser and navigate to the local URL provided by Streamlit.





