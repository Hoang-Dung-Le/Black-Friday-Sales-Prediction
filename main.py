import streamlit as st
import pickle

main_bg = "sample.jpg"
main_bg_ext = "jpg"

side_bg = "sample.jpg"
side_bg_ext = "jpg"

st.markdown(
    """
    <style>
        .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("BLACK FRIDAY SALE PREDICTION")

col1, col2 = st.columns([1, 1])

gender = col1.selectbox(label="Gender:", options=['M', 'F'])
age = col1.selectbox(label='Age:', options=['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])
occupation = col1.number_input(label='Occupation', min_value=0, max_value=20)
city_category = col1.selectbox(label='City_Category:', options=['A', 'B', 'C'])
stay_in_current_city_years = col2.selectbox(label='Stay_In_Current_City_Years:', options=['1', '2', '3','4+'])
marital_status = col2.selectbox(label='Marital_Status', options=['0', '1'])
product_category_1 = col2.number_input('product_category_1:', min_value=0, max_value=20)
product_category_2 = col2.number_input('product_category_2:', min_value=0, max_value=20)
product_category_3 = col2.number_input('product_category_3:', min_value=0, max_value=20)
model_selection = col1.selectbox(label='Model selection:', options=['Light Gradient Boosting Machine', 'Decision tree', 'Linear Regression'])

model = pickle.load(open('model.pickle', 'rb'))
model_ln = pickle.load(open('model_ln.pickle', 'rb'))
model_ds = pickle.load(open('model_decision_tree.pickle', 'rb'))
ct = pickle.load(open('ct.pickle', 'rb'))
min_max_scaler = pickle.load(open('min_max_scaler.pickle', 'rb'))

if stay_in_current_city_years == '4+':
    stay_in_current_city_years = 4
stay_in_current_city_years = float(stay_in_current_city_years)

submit = col1.button(label='Submit')

if submit:
    sample_test = [[gender, age, occupation, city_category, stay_in_current_city_years, int(marital_status), product_category_1,
         product_category_2, product_category_3]]
    sample_test = ct.transform(sample_test)
    sample_test = min_max_scaler.transform(sample_test)
    if model_selection == 'Light Gradient Boosting Machine':
        col2.write('Predict: ')
        col2.text(model.predict(sample_test)[0])
    elif model_selection == 'Decision tree':
        col2.write('Predict: ')
        col2.text(model_ds.predict(sample_test)[0])
    else :
        col2.write('Predict: ')
        col2.text(model_ln.predict(sample_test)[0])