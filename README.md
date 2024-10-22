# Hacktoberfest24-Mushroom-Edibility-Prediction-ML 🚀

This project is a sincere attempt by MSTC, DA-IICT to encourage Open Source contribution. Make the best out of the ongoing Hacktoberfest 2024 by contributing to for-the-community projects. This project participates in Hacktoberest 2024 and all successful PRs made here which is in accordance with hacktoberfest [guidelines](https://hacktoberfest.com/participation/#pr-mr-details) will be counted, and you have to make at least 4 successful pull requests in order to be eligible for the Hacktoberfest appreciation (Digital Rewards).


<img src="https://res.cloudinary.com/dbvyvfe61/image/upload/v1619799241/Cicada%203301:%20Reinvented/MSTC_ffmo9v.png" width="10%">


## Mushroom Edibility Prediction App 🍄
###### A Machine Learning project for Hactoberfest 2024, maintained by MSTC DA-IICT.

This Streamlit-based web application predicts whether a mushroom is edible or poisonous based on a set of biological features. It uses a trained Random Forest Classifier to make predictions from user inputs.

### Features
- **Cap characteristics**: Shape, surface, color
- **Bruises**: Presence or absence of bruising
- **Odor**: Various mushroom odors
- **Gill details**: Attachment, spacing, size, color
- **Stalk properties**: Shape, root, surface, color
- **Veil characteristics**: Type and color
- **Other attributes**: Ring number, ring type, spore print color, population, and habitat

### How to Run the App Locally

#### 1. Clone the Repository

```bash
git clone https://github.com/yusufokunlola/Hacktoberfest24-Mushroom-Edibility-Prediction-ML

cd Hacktoberfest24-Mushroom-Edibility-Prediction-ML
```

#### 2. Install Dependencies

Ensure that Python and pip are installed on your machine, then run:

```bash
pip install -r requirements.txt
```

#### 3. Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser.

### Dataset

The dataset used for training the Random Forest model consists of biological features of mushrooms, including:
- Cap shape, surface, color
- Bruises, odor
- Gill attachment, spacing, size, color
- Stalk shape, root, surface, color
- Veil type, color
- Ring number, type
- Spore print color
- Population, habitat

The target variable is the mushroom's edibility: either `edible (0)` or `poisonous (1)`.

### App Overview

- The app's sidebar provides information about the key features used for prediction.
- The main page allows users to input various mushroom attributes to predict its edibility.
  
### Prediction

- Users can input characteristics such as cap shape, gill size, stalk surface, and more.
- The trained Random Forest Classifier model will then predict whether the mushroom is edible or poisonous.
- Results are displayed with a user-friendly interface.

### Libraries Used

- `Pandas`: For data manipulation.
- `Scikit-learn`: For model training and prediction.
- `Pickle`: For saving and loading the trained model.
- `Streamlit`: For creating the web interface.

### Model Training

- **Algorithm**: Random Forest Classifier
- **Data Split**: 70% for training, 30% for testing
- **Performance**: The model has been trained on a cleaned and preprocessed dataset, and it is pickled for reuse in the app.

### File Structure

- `app.py`: Main script that runs the Streamlit application.
- `dataset/mushroom.csv`: The dataset used for training and testing the model.
- `rf_model.pkl`: Pre-trained Random Forest Classifier model saved using pickle.

### 🔗 Connect MSTC DA-IICT
Get in touch with us on [LinkedIn](https://www.linkedin.com/company/microsoft-student-technical-club-da-iict) / [Instagram](https://www.instagram.com/mstc_daiict)

Any query? Write to us at microsoftclub@daiict.ac.in

[<img src = "https://user-images.githubusercontent.com/112422657/193991648-3b62790c-a1e9-461e-9dff-e93bd045c06d.png" width = "200" />](https://github.com/MSTC-DA-IICT/Hacktoberfest24-Mushroom-Edibility-Prediction-ML)