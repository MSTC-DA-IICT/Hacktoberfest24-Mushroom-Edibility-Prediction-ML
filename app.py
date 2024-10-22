# import libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Mushroom Edibility Prediction App", page_icon="🍄") 
    
# Sidebar for app information
st.sidebar.title('App Information')
st.sidebar.write("""
    This app predicts mushroom edibility based on the following key features:
    - Cap characteristics (shape, surface, color)
    - Bruises, Odor
    - Gill details (attachment, spacing, size, color)
    - Stalk properties (shape, root, surface, color)
    - Veil color, Ring number, Ring type
    - Spore print color, Population, Habitat
""")
st.sidebar.write("Please enter the mushroom features on the main page to get the prediction.")

st.text('')
st.text('')
st.sidebar.markdown('`Code:` [GitHub](https://github.com/yusufokunlola/Hacktoberfest24-Mushroom-Edibility-Prediction-ML/)')

# Title and description
st.title("Mushroom Edibility Prediction App 🍄")

# Data cleaning function
def wrangle_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # fill missing values with 'unknown'
    df['stalk-root'].fillna('unknown')
    
    # drop the 'Unnamed: 0' column and 'veil-type' which contains a single value
    df.drop(columns=['Unnamed: 0', 'veil-type'], inplace=True)
    
    # map target variable edible as 0, poisonous as 1
    df['poisonous'] = df['poisonous'].map({'e': 0, 'p': 1})
    
    # Drop duplicates if exists
    df = df.drop_duplicates()

    return df

# label encoding and categorical mapping function
def preprocess_data(df):
    # Encoding categorical variables
    label_encoders = {}
    category_mappings = {}
    
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
        # mapping of the original category to the encoded value
        category_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    return df, label_encoders, category_mappings

# specify the data path and implement the wrangle function to clean the data
data_path = "dataset/mushroom.csv"
df = wrangle_data(data_path)

# preprocess data
df, label_encoders, category_mappings = preprocess_data(df)

# copy the processed dataframe
df = df.copy()


# Model Development
# create explanatory variable (X) and response variable (y)
X = df.drop(columns=['poisonous'])
y = df['poisonous']

# split data into train and test sets - 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create an instance of the random forest classifier and train on the train set
clf = RandomForestClassifier().fit(X_train, y_train)

#  save pickle file
pickle.dump(clf, open('rf_model.pkl','wb'))

# load model
rf_classif = pickle.load(open('rf_model.pkl','rb'))


# Create user-friendly interface
# Create two columns for the input features
col1, col2, col3 = st.columns(3)

# Input features in two columns
with col1:
    cap_shape = st.selectbox('Cap Shape', options=list(category_mappings['cap-shape'].keys()))
    cap_surface = st.selectbox('Cap Surface', options=list(category_mappings['cap-surface'].keys()))
    cap_color = st.selectbox('Cap Color', options=list(category_mappings['cap-color'].keys()))
    bruises = st.selectbox('Bruises', options=list(category_mappings['bruises'].keys()))
    odor = st.selectbox('Odor', options=list(category_mappings['odor'].keys()))
    gill_attachment = st.selectbox('Gill Attachment', options=list(category_mappings['gill-attachment'].keys()))
    gill_spacing = st.selectbox('Gill Spacing', options=list(category_mappings['gill-spacing'].keys()))
    
with col2:
    gill_size = st.selectbox('Gill Size', options=list(category_mappings['gill-size'].keys()))
    gill_color = st.selectbox('Gill Color', options=list(category_mappings['gill-color'].keys()))
    stalk_shape = st.selectbox('Stalk Shape', options=list(category_mappings['stalk-shape'].keys()))
    stalk_root = st.selectbox('Stalk Root', options=list(category_mappings['stalk-root'].keys()))
    stalk_surface_above_ring = st.selectbox('Stalk Surface Above Ring', options=list(category_mappings['stalk-surface-above-ring'].keys()))
    stalk_surface_below_ring = st.selectbox('Stalk Surface Below Ring', options=list(category_mappings['stalk-surface-below-ring'].keys()))
    stalk_color_above_ring = st.selectbox('Stalk Color Above Ring', options=list(category_mappings['stalk-color-above-ring'].keys()))
       
with col3: 
    stalk_color_below_ring = st.selectbox('Stalk Color Below Ring', options=list(category_mappings['stalk-color-below-ring'].keys()))
    veil_color = st.selectbox('Veil Color', options=list(category_mappings['veil-color'].keys()))
    ring_number = st.selectbox('Ring Number', options=list(category_mappings['ring-number'].keys()))
    ring_type = st.selectbox('Ring Type', options=list(category_mappings['ring-type'].keys()))
    spore_print_color = st.selectbox('Spore Print Color', options=list(category_mappings['spore-print-color'].keys()))
    population = st.selectbox('Population', options=list(category_mappings['population'].keys()))
    habitat = st.selectbox('Habitat', options=list(category_mappings['habitat'].keys()))

# Encoding categorical features
cap_shape_encoded = category_mappings['cap-shape'][cap_shape]
cap_surface_encoded = category_mappings['cap-surface'][cap_surface]
cap_color_encoded = category_mappings['cap-color'][cap_color]
bruises_encoded = category_mappings['bruises'][bruises]
odor_encoded = category_mappings['odor'][odor]
gill_attachment_encoded = category_mappings['gill-attachment'][gill_attachment]
gill_spacing_encoded = category_mappings['gill-spacing'][gill_spacing]
gill_size_encoded = category_mappings['gill-size'][gill_size]
gill_color_encoded = category_mappings['gill-color'][gill_color]
stalk_shape_encoded = category_mappings['stalk-shape'][stalk_shape]
stalk_root_encoded = category_mappings['stalk-root'][stalk_root]
stalk_surface_above_ring_encoded = category_mappings['stalk-surface-above-ring'][stalk_surface_above_ring]
stalk_surface_below_ring_encoded = category_mappings['stalk-surface-below-ring'][stalk_surface_below_ring]
stalk_color_above_ring_encoded = category_mappings['stalk-color-above-ring'][stalk_color_above_ring]
stalk_color_below_ring_encoded = category_mappings['stalk-color-below-ring'][stalk_color_below_ring]
veil_color_encoded = category_mappings['veil-color'][veil_color]
ring_number_encoded = category_mappings['ring-number'][ring_number]
ring_type_encoded = category_mappings['ring-type'][ring_type]
spore_print_color_encoded = category_mappings['spore-print-color'][spore_print_color]
population_encoded = category_mappings['population'][population]
habitat_encoded = category_mappings['habitat'][habitat]

# Create the input DataFrame for prediction in the specified order
input_data = pd.DataFrame({
    'cap_shape': [cap_shape_encoded],
    'cap_surface': [cap_surface_encoded],
    'cap_color': [cap_color_encoded],
    'bruises': [bruises_encoded],
    'odor': [odor_encoded],
    'gill_attachment': [gill_attachment_encoded],
    'gill_spacing': [gill_spacing_encoded],
    'gill_size': [gill_size_encoded],
    'gill_color': [gill_color_encoded],
    'stalk_shape': [stalk_shape_encoded],
    'stalk_root': [stalk_root_encoded],
    'stalk_surface_above_ring': [stalk_surface_above_ring_encoded],
    'stalk_surface_below_ring': [stalk_surface_below_ring_encoded],
    'stalk_color_above_ring': [stalk_color_above_ring_encoded],
    'stalk_color_below_ring': [stalk_color_below_ring_encoded],
    'veil_color': [veil_color_encoded],
    'ring_number': [ring_number_encoded],
    'ring_type': [ring_type_encoded],
    'spore_print_color': [spore_print_color_encoded],
    'population': [population_encoded],
    'habitat': [habitat_encoded]
})

# Align the input features with the training features
input_data_aligned = input_data.reindex(columns=X_train.columns, fill_value=0)

# Prediction button
if st.button('Predict Mushroom Edibility'):
    prediction = rf_classif.predict(input_data_aligned)
    prediction_label = "Edible" if prediction[0] == 0 else "Poisonous"
    st.markdown(f"### Mushroom Edibility: <strong style='color:tomato;'>{prediction_label}</strong>", unsafe_allow_html=True)