import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Reading supply and demand data
supply_data = pd.read_csv(r"supply.csv")
demand_data = pd.read_csv(r"demand.csv")


# Convert 'DATE' columns to datetime to ensure proper date handling
supply_data['DATE'] = pd.to_datetime(supply_data['DATE'])
demand_data['DATE'] = pd.to_datetime(demand_data['DATE'])

# Sort dataframes by 'DATE' to ensure data is in chronological order
supply_data = supply_data.sort_values('DATE')
demand_data = demand_data.sort_values('DATE')

# Merge supply and demand data based on 'DATE' using suffixes to distinguish columns from each dataset
merged_data = pd.merge(supply_data, demand_data, on='DATE', suffixes=('_supply', '_demand'))

# Drop rows with missing values in specific columns to ensure data quality
# These columns are essential for the analysis
merged_data.dropna(subset=['MSACSR', 'PERMIT', 'TLRESCONS', 'EVACANTUSQ176N', 'MORTGAGE30US', 'GDP', 'UMCSENT'], inplace=True)

# Impute missing values in 'INTDSRUSM193N' with the mean to handle missing data
imputer = SimpleImputer(strategy='mean')
merged_data['INTDSRUSM193N'] = imputer.fit_transform(merged_data[['INTDSRUSM193N']])

# Reset the index for the cleaned and prepared dataset
merged_data = merged_data.reset_index(drop=True)

# Drop the 'CSUSHPISA_supply' column as it's no longer needed
merged_data.drop('CSUSHPISA_supply', axis=1, inplace=True)

# Rename the 'CSUSHPISA_demand' column to 'CSUSHPISA' for consistency
merged_data.rename(columns={'CSUSHPISA_demand': 'CSUSHPISA'}, inplace=True)

# Fill missing values in 'CSUSHPISA' with the mean of the column
merged_data['CSUSHPISA'] = merged_data['CSUSHPISA'].fillna(merged_data['CSUSHPISA'].mean())


# Define the features and target variable
features = ['MSACSR', 'PERMIT', 'TLRESCONS', 'EVACANTUSQ176N', 'MORTGAGE30US', 'GDP', 'UMCSENT', 'INTDSRUSM193N', 'MSPUS']
target = 'CSUSHPISA'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(merged_data[features], merged_data[target], test_size=0.2, random_state=42)

# Define a dictionary of regression models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression': SVR(),
    'Neural Network': MLPRegressor()
}

# Initialize a dictionary to store model evaluation results
results = {}

# Perform cross-validation and calculate mean squared error for each model
for model_name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores
    avg_mse = mse_scores.mean()
    results[model_name] = avg_mse

# Select the best model based on mean squared error
best_model = min(results, key=results.get)
best_model_instance = models[best_model]

# Fit the best model to the training data
best_model_instance.fit(X_train, y_train)

# Make predictions on the testing set
predictions = best_model_instance.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Import the R-squared score function
from sklearn.metrics import r2_score

# Calculate the R-squared score
r2 = r2_score(y_test, predictions)



# Fit the best model to the training data
best_model_instance.fit(X_train, y_train)

# Get the coefficients from the model
coefficients = best_model_instance.coef_


st.set_page_config(
    page_title="Housing Price Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
    
)

st.header("🏠 US Housing Price Analysis ")
st.markdown("""
**Created by:** Shivam Bahuguna  
**Created for:** Home.LLC             
**Date:** September 17, 2023
""")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Define a function to create and display the selected chart
def selected_chart():

    selected_chart = st.selectbox("👇Key Features", ["Select a Feature",
        "Monthly Supply vs. Home Price",
        "Vacant Housing Units vs. Home Price",
        "New Housing Units Authorized vs. Home Price",
        "Total Construction Spending vs. Home Price",
        "30-Year Fixed Rate Mortgage vs. Home Price",
        "GDP vs. Home Price",
        "Consumer Sentiment vs. Home Price",
        "Interest Rates vs. Home Price",
        "Median Sales Price vs. Home Price"
    ])

    if selected_chart != "Select a Feature" and st.button("Generate Chart"):


        merged_data['DATE'] = pd.to_datetime(merged_data['DATE'])
        merged_data.set_index('DATE', inplace=True)

        merged_data['MSACSR'] = pd.to_numeric(merged_data['MSACSR'], errors='coerce')
        merged_data['CSUSHPISA'] = pd.to_numeric(merged_data['CSUSHPISA'], errors='coerce')
        merged_data['PERMIT'] = pd.to_numeric(merged_data['PERMIT'], errors='coerce')
        merged_data['TLRESCONS'] = pd.to_numeric(merged_data['TLRESCONS'], errors='coerce')
        merged_data['EVACANTUSQ176N'] = pd.to_numeric(merged_data['EVACANTUSQ176N'], errors='coerce')
        merged_data['MORTGAGE30US'] = pd.to_numeric(merged_data['MORTGAGE30US'], errors='coerce')
        merged_data['GDP'] = pd.to_numeric(merged_data['GDP'], errors='coerce')
        merged_data['UMCSENT'] = pd.to_numeric(merged_data['UMCSENT'], errors='coerce')

        merged_data.dropna(subset=['MSACSR', 'PERMIT', 'TLRESCONS', 'EVACANTUSQ176N', 'MORTGAGE30US', 'GDP', 'UMCSENT' ], inplace=True)



        if selected_chart == "Monthly Supply vs. Home Price":
            merged_data['QUARTER'] = merged_data.index.to_period('Q')
            merged_data['QUARTER'] = merged_data['QUARTER'].astype(str)
            grouped_data = merged_data.groupby('QUARTER').agg({'MSACSR': 'sum', 'CSUSHPISA': 'mean'}).reset_index()
            scaler = MinMaxScaler()
            grouped_data[['MSACSR', 'CSUSHPISA']] = scaler.fit_transform(grouped_data[['MSACSR', 'CSUSHPISA']])
            grouped_data = grouped_data.sort_values('QUARTER')
            plt.figure(figsize=(16, 6))
            sns.barplot(x='QUARTER', y='MSACSR', data=grouped_data, color='skyblue', label='MSACSR')
            sns.lineplot(x='QUARTER', y='CSUSHPISA', data=grouped_data, marker='o', linestyle='-', color='red', label='CSUSHPISA')
            plt.xlabel('Quarter')
            plt.ylabel('Normalized Values')
            plt.title('Monthly Supply of New Houses vs CSUSHPISA (Normalized)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            st.pyplot(plt.gcf())
            st.subheader("Monthly Supply of New Houses vs. Home Price Index (CSUSHPISA)")
            st.write("The Monthly Supply of New Houses in the United States (MSACSR) represents the ratio of the current new for-sale inventory to the number of new houses being sold. It indicates how long the existing inventory would last at the current sales rate if no additional new houses were built. Importantly, MSACSR has a negative correlation with home prices, meaning that an increase in MSACSR may lead to a decrease in the S&P/Case-Shiller U.S. National Home Price Index. This relationship arises because an increase in the supply of new houses can potentially reduce demand, which in turn may result in lower home prices.")
        
        elif selected_chart == "Vacant Housing Units vs. Home Price":
            merged_data['QUARTER'] = merged_data.index.to_period('Q')
            merged_data['QUARTER'] = merged_data['QUARTER'].astype(str)
            grouped_data = merged_data.groupby('QUARTER').agg({'EVACANTUSQ176N': 'sum', 'CSUSHPISA': 'mean'}).reset_index()
            scaler = MinMaxScaler()
            grouped_data[['EVACANTUSQ176N', 'CSUSHPISA']] = scaler.fit_transform(grouped_data[['EVACANTUSQ176N', 'CSUSHPISA']])
            grouped_data = grouped_data.sort_values('QUARTER')
            plt.figure(figsize=(16, 6))
            sns.barplot(x='QUARTER', y='EVACANTUSQ176N', data=grouped_data, color='skyblue', label='Median Sales Price')
            sns.lineplot(x='QUARTER', y='CSUSHPISA', data=grouped_data, marker='o', linestyle='-', color='red', label='CSUSHPISA')
            plt.xlabel('Quarter')
            plt.ylabel('Normalized Values')
            plt.title('Vacant Housing Units vs CSUSHPISA (Normalized)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            st.subheader("Vacant Housing Units vs. Home Price Index (CSUSHPISA)")
            st.write("""Imagine there are houses for sale, and some of them are empty and waiting for buyers. When there are not many empty houses (low inventory), home prices tend to go up because buyers compete and offer more money to buy a home. It's like a bidding war.
    But when there are lots of empty houses (high inventory), it can push home prices down. That's because there are more houses than buyers, so sellers might lower prices to attract buyers. So, the number of empty houses can affect whether home prices go up or down.""")
        
        elif selected_chart == "New Housing Units Authorized vs. Home Price":
            merged_data['QUARTER'] = merged_data.index.to_period('Q')
            merged_data['QUARTER'] = merged_data['QUARTER'].astype(str)
            grouped_data = merged_data.groupby('QUARTER').agg({'PERMIT': 'sum', 'CSUSHPISA': 'mean'}).reset_index()
            scaler = MinMaxScaler()
            grouped_data[['PERMIT', 'CSUSHPISA']] = scaler.fit_transform(grouped_data[['PERMIT', 'CSUSHPISA']])
            grouped_data = grouped_data.sort_values('QUARTER')
            plt.figure(figsize=(16, 6))
            sns.barplot(x='QUARTER', y='PERMIT', data=grouped_data, color='skyblue', label='Median Sales Price')
            sns.lineplot(x='QUARTER', y='CSUSHPISA', data=grouped_data, marker='o', linestyle='-', color='red', label='CSUSHPISA')
            plt.xlabel('Quarter')
            plt.ylabel('Normalized Values')
            plt.title('New Housing Units Authorized vs CSUSHPISA (Normalized)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            # Pass the Matplotlib figure to st.pyplot
            st.pyplot(plt.gcf())
            st.subheader("New Housing Units Authorized vs. Home Price Index (CSUSHPISA)")
            st.write("New Privately-Owned Housing Units Authorized (PERMIT) counts how many new houses are allowed to be built. When more permits are given (high PERMIT), it's a good sign for home prices. It means there's demand, and prices can go up, like when everyone wants a limited-edition toy.")

        elif selected_chart == "Total Construction Spending vs. Home Price":
            merged_data['QUARTER'] = merged_data.index.to_period('Q')
            merged_data['QUARTER'] = merged_data['QUARTER'].astype(str)
            grouped_data = merged_data.groupby('QUARTER').agg({'TLRESCONS': 'sum', 'CSUSHPISA': 'mean'}).reset_index()
            scaler = MinMaxScaler()
            grouped_data[['TLRESCONS', 'CSUSHPISA']] = scaler.fit_transform(grouped_data[['TLRESCONS', 'CSUSHPISA']])
            grouped_data = grouped_data.sort_values('QUARTER')
            plt.figure(figsize=(16, 6))
            sns.barplot(x='QUARTER', y='TLRESCONS', data=grouped_data, color='skyblue', label='Median Sales Price')
            sns.lineplot(x='QUARTER', y='CSUSHPISA', data=grouped_data, marker='o', linestyle='-', color='red', label='CSUSHPISA')
            plt.xlabel('Quarter')
            plt.ylabel('Normalized Values')
            plt.title('Construction Spending vs CSUSHPISA (Normalized)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            st.subheader("Total Construction Spending vs. Home Price Index (CSUSHPISA)")
            st.write("Total Construction Spending on Homes is like a measure of how much money is spent on building houses in the U.S. When this spending is high, it usually means good news for home prices. More home building often leads to higher demand, which can drive up prices, like a popular item getting more expensive")

        elif selected_chart == "30-Year Fixed Rate Mortgage vs. Home Price":
            merged_data['QUARTER'] = merged_data.index.to_period('Q')
            merged_data['QUARTER'] = merged_data['QUARTER'].astype(str)
            grouped_data = merged_data.groupby('QUARTER').agg({'MORTGAGE30US': 'sum', 'CSUSHPISA': 'mean'}).reset_index()
            scaler = MinMaxScaler()
            grouped_data[['MORTGAGE30US', 'CSUSHPISA']] = scaler.fit_transform(grouped_data[['MORTGAGE30US', 'CSUSHPISA']])
            grouped_data = grouped_data.sort_values('QUARTER')
            plt.figure(figsize=(16, 6))
            sns.barplot(x='QUARTER', y='MORTGAGE30US', data=grouped_data, color='skyblue', label='Median Sales Price')
            sns.lineplot(x='QUARTER', y='CSUSHPISA', data=grouped_data, marker='o', linestyle='-', color='red', label='CSUSHPISA')
            plt.xlabel('Quarter')
            plt.ylabel('Normalized Values')
            plt.title('30-Year Fixed Rate Mortgag vs CSUSHPISA (Normalized)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            st.subheader("30-Year Fixed Rate Mortgage vs. Home Price Index (CSUSHPISA)")
            st.write("The 30-Year Fixed Rate Mortgage Average in the United States is like a gauge for mortgage interest rates. When rates are low, it's easier for folks to afford homes, boosting demand and raising home prices. Conversely, high rates make buying homes tough, reducing demand and lowering prices. Keep in mind that this chart lags behind, so it might not show an immediate connection between the variables.")

        elif selected_chart == "GDP vs. Home Price":
            merged_data['QUARTER'] = merged_data.index.to_period('Q')
            merged_data['QUARTER'] = merged_data['QUARTER'].astype(str)
            grouped_data = merged_data.groupby('QUARTER').agg({'GDP': 'sum', 'CSUSHPISA': 'mean'}).reset_index()
            scaler = MinMaxScaler()
            grouped_data[['GDP', 'CSUSHPISA']] = scaler.fit_transform(grouped_data[['GDP', 'CSUSHPISA']])
            grouped_data = grouped_data.sort_values('QUARTER')
            plt.figure(figsize=(16, 6))
            sns.barplot(x='QUARTER', y='GDP', data=grouped_data, color='skyblue', label='Median Sales Price')
            sns.lineplot(x='QUARTER', y='CSUSHPISA', data=grouped_data, marker='o', linestyle='-', color='red', label='CSUSHPISA')
            plt.xlabel('Quarter')
            plt.ylabel('Normalized Values')
            plt.title('GDP vs CSUSHPISA (Normalized)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            st.subheader("GDP vs. Home Price Index (CSUSHPISA)")
            st.write("The Gross Domestic Product (GDP) is like a measure of the overall economic health of a country. When GDP is strong, it often means a healthy economy, and that can boost home prices. The chart shows that when GDP goes up (more economic activity), home prices tend to follow suit. It's like when people have more money, they're more likely to buy homes, which can push prices up.")

        elif selected_chart == "Consumer Sentiment vs. Home Price":
            merged_data['QUARTER'] = merged_data.index.to_period('Q')
            merged_data['QUARTER'] = merged_data['QUARTER'].astype(str)
            grouped_data = merged_data.groupby('QUARTER').agg({'UMCSENT': 'sum', 'CSUSHPISA': 'mean'}).reset_index()
            scaler = MinMaxScaler()
            grouped_data[['UMCSENT', 'CSUSHPISA']] = scaler.fit_transform(grouped_data[['UMCSENT', 'CSUSHPISA']])
            grouped_data = grouped_data.sort_values('QUARTER')
            plt.figure(figsize=(16, 6))
            sns.barplot(x='QUARTER', y='UMCSENT', data=grouped_data, color='skyblue', label='Median Sales Price')
            sns.lineplot(x='QUARTER', y='CSUSHPISA', data=grouped_data, marker='o', linestyle='-', color='red', label='CSUSHPISA')
            plt.xlabel('Quarter')
            plt.ylabel('Normalized Values')
            plt.title('Consumer Sentiment vs CSUSHPISA (Normalized)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            st.subheader("Consumer Sentiment vs. Home Price Index (CSUSHPISA)")
            st.write("The University of Michigan Consumer Sentiment Index reflects how confident people feel about the economy. When confidence is high, folks tend to spend more, including on big purchases like homes. This increased demand for homes can drive up prices, as shown in the chart")

        elif selected_chart == "Interest Rates vs. Home Price":
            merged_data['QUARTER'] = merged_data.index.to_period('Q')
            merged_data['QUARTER'] = merged_data['QUARTER'].astype(str)
            grouped_data = merged_data.groupby('QUARTER').agg({'INTDSRUSM193N': 'sum', 'CSUSHPISA': 'mean'}).reset_index()
            scaler = MinMaxScaler()
            grouped_data[['INTDSRUSM193N', 'CSUSHPISA']] = scaler.fit_transform(grouped_data[['INTDSRUSM193N', 'CSUSHPISA']])
            grouped_data = grouped_data.sort_values('QUARTER')
            plt.figure(figsize=(16, 6))
            sns.barplot(x='QUARTER', y='INTDSRUSM193N', data=grouped_data, color='skyblue', label='Median Sales Price')
            sns.lineplot(x='QUARTER', y='CSUSHPISA', data=grouped_data, marker='o', linestyle='-', color='red', label='CSUSHPISA')
            plt.xlabel('Quarter')
            plt.ylabel('Normalized Values')
            plt.title('Interest Rates vs CSUSHPISA (Normalized)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(plt.gcf())        
            st.subheader("Interest Rates vs. Home Price Index (CSUSHPISA)")
            st.write("The Interest Rates and Discount Rate are tools used by the Federal Reserve to control the supply of money in the economy. Lower interest rates make it easier for people to borrow money, increasing demand for homes and driving up prices, as seen in the chart.")

        elif selected_chart == "Median Sales Price vs. Home Price":
            merged_data['QUARTER'] = merged_data.index.to_period('Q')
            merged_data['QUARTER'] = merged_data['QUARTER'].astype(str)
            grouped_data = merged_data.groupby('QUARTER').agg({'MSPUS': 'sum', 'CSUSHPISA': 'mean'}).reset_index()
            scaler = MinMaxScaler()
            grouped_data[['MSPUS', 'CSUSHPISA']] = scaler.fit_transform(grouped_data[['MSPUS', 'CSUSHPISA']])
            grouped_data = grouped_data.sort_values('QUARTER')
            plt.figure(figsize=(16, 6))
            sns.barplot(x='QUARTER', y='MSPUS', data=grouped_data, color='skyblue', label='Median Sales Price')
            sns.lineplot(x='QUARTER', y='CSUSHPISA', data=grouped_data, marker='o', linestyle='-', color='red', label='CSUSHPISA')
            plt.xlabel('Quarter')
            plt.ylabel('Normalized Values')
            plt.title('Median Sales Price vs CSUSHPISA (Normalized)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            st.subheader("Median Sales Price vs. Home Price Index (CSUSHPISA)")
            st.write("The Median Sales Price reflects the midpoint of house sale prices, indicating that half sold for more and half for less. A rise in median prices is closely linked to higher home prices.")
   
selected_chart()




def model():
    models = {
        'Select a Model': None,  # Initial placeholder with no model selected
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Support Vector Regression': SVR(),
        'Neural Network': MLPRegressor()
    }


    # Create a Streamlit selectbox to choose the regression model
    selected_model = st.selectbox("👇Regression Models", list(models.keys()))

    # Initialize a dictionary to store model evaluation results
    results = {}
    # Check if a valid model is selected (not the placeholder)


    if selected_model != 'Select a Model':
        # Perform cross-validation and calculate mean squared error for the selected model
        if models != "Select a Feature" and st.button("Generate Output"):
            
            model_instance = models[selected_model]
            scores = cross_val_score(model_instance, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            mse_scores = -scores
            avg_mse = mse_scores.mean()
            results[selected_model] = avg_mse

        # Fit the selected model to the training data
            model_instance.fit(X_train, y_train)

        # Make predictions on the testing set
            predictions = model_instance.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

        # Display the results if a model is selected
            st.subheader("Model Output")
            if selected_model == 'Linear Regression':
                st.markdown("<span style='color: red;'>**Best Model**</span>", unsafe_allow_html=True)

            st.write("Model Selection Results:")
            for model, mse_score in results.items():
                st.write(f"{model}: MSE=<span style='color: red;'>{mse_score}</span>", unsafe_allow_html=True)


        # Display information about the selected model
            st.subheader("Selected Model Information")
            st.write(f"Selected Model: {selected_model}")
            st.write(f"Selected Model MSE on Testing Set: <span style='color: red;'>{mse}</span>", unsafe_allow_html=True)

        # Display model coefficients if the model is linear regression
            if selected_model == 'Linear Regression':
                coefficients = model_instance.coef_
                st.subheader("R-squared score")
                st.write(f"R-squared score: <span style='color: red;'>{r2}</span>", unsafe_allow_html=True)
                st.subheader("Model Coefficients")
                st.write("Coefficients:")
                for feature, coefficient in zip(features, coefficients):
                    st.write(f"{feature}: <span style='color: red;'>{coefficient}</span>", unsafe_allow_html=True)
model()

st.sidebar.header("Details:")

if st.sidebar.button("Data Source"):
    # Set the button_clicked variable to True when the button is clicked
    button_clicked = True

    st.markdown(""" 
#### Data Source            

he following data sources will be utilized for this analysis:

- [S&P Case-Shiller Home Price Index (CSUSHPISA)](https://fred.stlouisfed.org/series/CSUSHPISA): This index will serve as a proxy for home prices in the United States.

- Other publicly available datasets will be sourced individually for the key factors that influence home prices nationally. These factors may include but are not limited to:
  - Mortgage rates
  - Unemployment rates
  - GDP growth
  - Housing supply and demand data
  - Housing affordability indexes
  - Consumer sentiment
  - Building permits and construction data
  - Demographic data (population growth, age distribution, etc.)
                
#### Data Collection

##### Supply Data
- **DATE**: The date of the observation (2003 - 2023).
- **CSUSHPISA**: S&P/Case-Shiller U.S. National Home Price Index.
- **MSACSR**: Monthly Supply of New Houses in the United States.
- **PERMIT**: Represents the number of new housing units authorized for construction in permit-issuing places (in thousands of units).
- **TLRESCONS**: Represents the total construction spending on residential projects (in millions of dollars).
- **EVACANTUSQ176N**: Provides an estimate of the number of vacant housing units in the United States (in thousands of units).


##### Demand Data
- **DATE**: The date of the observation (2003 - 2023).
- **CSUSHPISA**: Serves as a proxy for home prices and represents the home price index for the United States.
- **MORTGAGE15US**: 30-Year Fixed Rate Mortgage Average in the United States (in percent).
- **UMCSENT**: Measures the consumer sentiment index based on surveys conducted by the University of Michigan.
- **INTDSRUSM193N**: Represents the interest rates or discount rates for the United States (in billions of dollars).
- **MSPUS**: Median Sales Price of Houses Sold for the United States.
- **GDP**: Gross Domestic Product (in billions of dollars).               
""")

if st.sidebar.button("Model Evaluation"):
    # Set the button_clicked variable to True when the button is clicked
    button_clicked = True

    st.markdown("""
#### Model Evaluation
                
We assessed our model's performance using mean squared error (MSE) and the R-squared score. MSE gauges prediction accuracy, with lower values indicating better performance. Our model achieved an MSE of 33.18 on the test data, signifying low prediction errors. The R-squared score, measuring how well features explain the target's variance, yielded a value of 0.9723, implying a strong fit.
                
**Key Coefficient Insights (scaled data):**

- 'PERMIT' (new housing units authorized) had a small positive coefficient (0.0197), indicating a weak positive link with home prices.

- 'MSACSR' (monthly new house supply) had a positive coefficient (8.17), suggesting that more monthly new houses correlate with higher home prices.

- 'TLRESCONS' (construction spending on residential projects) showed a positive coefficient (5.693), signifying a minimal impact on home prices.

- 'EVACANTUSQ176N' (estimated vacant housing units) had a negative coefficient (-0.00133), indicating more vacant units related to slightly lower home prices.

- 'MORTGAGE30US' (30-year fixed-rate mortgage) exhibited a negative coefficient (-14.994), implying higher rates linked to lower home prices.

- 'GDP' (Gross Domestic Product) had a very small negative coefficient (-0.00303), suggesting higher GDP relates to lower home prices.

- 'UMCSENT' (consumer sentiment) showed a negative coefficient (-0.18699), meaning lower sentiment associated with lower home prices.

- 'INTDSRUSM193N' (interest or discount rates) had a positive coefficient (3.97), suggesting higher rates linked to higher home prices.

- 'MSPUS' (median sales price) had a small positive coefficient (0.000455), indicating a weak positive link with home prices.

- These coefficients help understand feature importance and their impact on home prices


**Summary:**

Our Linear Regression model performed well with low MSE and a high R-squared score. Coefficients provide insights into feature importance and direction of influence on home prices.
""")
    
if st.sidebar.button("Key Findings"):
    # Set the button_clicked variable to True when the button is clicked
    button_clicked = True

    st.markdown("""
#### Key Findings               

**Supply Factors:**

- Monthly supply of new homes (MSACSR) has a minor negative influence on housing prices.
- A greater number of authorized housing units (PERMIT) is associated with higher property   prices.
- Total construction spending on residential projects (TLRESCONS) strongly correlates with higher home prices.
- More vacant housing units (EVACANTUSQ176N) may lower home prices.

**Demand Factors:**

- Higher mortgage rates (MORTGAGE30US) are slightly associated with lower home prices.
- Lower consumer sentiment (UMCSENT) is linked to slightly lower home prices.
- Higher interest or discount rates (INTDSRUSM193N) are negatively correlated with home prices.
- Stronger Gross Domestic Product (GDP) and higher median sales prices (MSPUS) strongly correlate with higher home prices.


**Insights:**

- Supply factors and strong economic indicators generally lead to higher home prices.
- Demand factors like higher mortgage rates and lower consumer sentiment can slightly reduce home prices.
- Economic factors such as GDP and interest rates significantly impact home prices.
- Market dynamics and buyer behavior, reflected in median sales prices, strongly influence home prices.""")


if st.sidebar.button("Conclusion"):
    # Set the button_clicked variable to True when the button is clicked
    button_clicked = True

    st.markdown("""
                
#### Conclusion
                                
More Homes, Higher Prices: When there are more new homes getting authorized for construction ('PERMIT') and a greater monthly supply of houses ('MSACSR'), it tends to push home prices up. It's like a supply and demand game.

- Build It, Prices Rise: Spending more on building residential projects ('TLRESCONS') generally leads to higher home prices. Why? Because building costs, like materials and labor, also rise.

- Vacant Homes and Rates: If there are more vacant houses ('EVACANTUSQ176N'), it might bring home prices down slightly. Higher mortgage rates ('MORTGAGE30US') can also lower prices as they make buying homes more expensive.

- Economic Powers: A strong economy, with a higher Gross Domestic Product (GDP), often means higher home prices. However, some interest rates ('INTDSRUSM193N') can increase prices.

- Sales Prices Matter: The median price of houses sold ('MSPUS') has a big impact. When it's higher, home prices tend to follow suit.

These insights can help everyone in the real estate world, from buyers and sellers to builders and policymakers, make smarter decisions. Understanding these factors is like having a secret formula for the housing market.
""")

# Disable the Streamlit onboarding sidebar
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showfileUploaderEncoding', False)
