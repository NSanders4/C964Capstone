#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the marketing data from CSV file
marketing_data = pd.read_csv('digital_marketing_campaign_dataset.csv')

# Drop AdvertisingPlatform, AdvertisingTool and CustomerID columns from the dataset
# These columns are removed as they are not relevant to the model.
marketing_data = marketing_data.drop(['AdvertisingPlatform', 'AdvertisingTool', 'CustomerID'], axis=1)

# Display the first few rows of the dataset to get an overview
marketing_data.head()

# Function to remove null, NaN and empty values from the dataset.
def remove_null_nan_empty(data):
    data = data.dropna()
    data = data.replace(r'^\s*$', np.nan, regex=True)
    data = data.dropna()
    return data

# Apply the function to the dataset
marketing_data = remove_null_nan_empty(marketing_data)

# Display the info of the dataset to determine object types and non-null counts
marketing_data.info()

# Display summary statistics for the dataset
marketing_data.describe()


# In[2]:


# Exclude the 'Conversion' column
columns_to_plot = marketing_data.columns.difference(['Conversion'])
subset_data = marketing_data[columns_to_plot]

# Plot histograms for numerical variables with increased spacing and rotated x-axis labels for better readability
subset_data.hist(bins=30, figsize=(22, 20))
plt.show()


# In[3]:


# Bar plots for categorical variables
plt.figure(figsize=(10, 6))
sns.countplot(x="CampaignChannel", hue="Conversion", data= marketing_data)
plt.legend(title="Conversion", loc="best")
plt.title("Campaign Channel vs Conversion")
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x="Gender", hue="Conversion", data=marketing_data)
plt.legend(title="Conversion", loc="best")
plt.title("Gender vs Conversion")
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x="CampaignType", hue="Conversion", data=marketing_data)
plt.legend(title="Conversion", loc="best")
plt.title("Campaign Type vs Conversion")
plt.show()


# In[4]:


# Correlation matrix for numerical features
corr_matrix = marketing_data[['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate', 'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'PreviousPurchases', 'LoyaltyPoints', 'Conversion']].corr()
plt.figure(figsize=(12, 10))
plt.title('Correlation Matrix For Numerical Features')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.3f')
plt.show()


# In[5]:


# Plot for the count of conversions
sns.countplot(x='Conversion', data= marketing_data)
plt.title('Conversion Count') 
plt.show() 


# In[6]:


# Identify categorical columns for encoding
categorical_columns = ["Gender", "CampaignChannel", "CampaignType"]

# Create a ColumnTransformer for one-hot encoding
# This will transform categorical variables into binary columns
ct = ColumnTransformer(
    [("encoder", OneHotEncoder(drop="first"), categorical_columns)],
    remainder="passthrough",
)

# Fit the ColumnTransformer to the data and transform it
encoded_data = ct.fit_transform(marketing_data)

# Get the feature names after encoding
feature_names = ct.get_feature_names_out()

# Clean up the feature names by removing the prefix
feature_names_cleaned = [name.split("__")[-1] for name in feature_names]

# Create a new DataFrame with the encoded data using the cleaned feature names
encoded_df = pd.DataFrame(encoded_data, columns=feature_names_cleaned)


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

# Separate features and target variable
X = encoded_df.drop(["Conversion"], axis=1)
y = encoded_df["Conversion"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the random forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict the test set
y_pred = rf_model.predict(X_test)

# Show detailed classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Generate a heatmap for the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()


# In[8]:


# Extract feature importances from the trained random forest model
importances = rf_model.feature_importances_

# Create a DataFrame to store feature names and their importances and sort 
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Bar chart of feature importances
plt.figure(figsize=(12, 8))  
sns.barplot(x='Importance', y='Feature', data=feature_importances) 
plt.title('Feature Importances from Random Forest')  
plt.show()  


# In[12]:


# Train the random forest classifier with class_weight='balanced' to account for imbalanced classes
rf_model_balanced = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_model_balanced.fit(X_train, y_train)

# Predict the test set
y_pred_balanced = rf_model_balanced.predict(X_test)

# Display the results
print("\nResults with Class Weight 'Balanced':")

# Show detailed classification report
print("\nClassification Report (Balanced):\n")
print(classification_report(y_test, y_pred_balanced))

# Generate a heatmap for the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_balanced)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()


# In[10]:


from sklearn.metrics import precision_recall_curve

# Get predicted probabilities
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Compute precision and recall for different thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob, pos_label=1)

plt.figure(figsize=(10, 7))

for i in range(0, len(thresholds), 10):  # Annotate every 10th threshold
    plt.annotate(
        f"{thresholds[i]:.2f}",
        (recalls[i], precisions[i]),
        textcoords="offset points",
        ha="center",
        va="top",
        xytext=(12, 15)
    )

# Plot the Precision-Recall curve
plt.plot(recalls, precisions, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve For Random Forest")
plt.legend(labels=['Thresholds'])
plt.show()

# Find the optimal threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Make predictions using the optimal threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)


# Display the results
print("\nResults with Optimal Threshold:")

# Evaluate the model
print(classification_report(y_test, y_pred_optimal))

# Generate a heatmap for the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()


# In[13]:


import ipywidgets as widgets
from IPython.display import display

# Create 5 marketing campaigns
campaigns = pd.DataFrame(
    {
        "CampaignName": [
            "Campaign 1",
            "Campaign 2",
            "Campaign 3",
            "Campaign 4",
            "Campaign 5",
        ],
        "CampaignChannel": ["PPC", "SEO", "Social Media", "Referral", "PPC"],
        "CampaignType": [
            "Conversion",
            "Consideration",
            "Retention",
            "Conversion",
            "Retention",
        ],
        "AdSpend": [7000, 5000, 3000, 9000, 4500],
        "ClickThroughRate": [0.28, 0.22, 0.08, 0.29, 0.16],
        "ConversionRate": [0.18, 0.15, 0.07, 0.18, 0.12],
        "WebsiteVisits": [45, 35, 20, 49, 30],
        "PagesPerVisit": [7.8, 6.0, 4.0, 9.5, 6.5],
        "TimeOnSite": [12.5, 8.5, 5.0, 14.5, 9.0],
        "SocialShares": [75, 60, 40, 95, 55],
        "EmailOpens": [18, 12, 8, 18, 10],
        "EmailClicks": [8, 5, 3, 7, 5],
    }
)



# Input widgets for customer data 
age_input = widgets.IntText(
    description="Age:", value=30, layout=widgets.Layout(width="300px")
)
gender_input = widgets.Dropdown(
    options=["Male", "Female"],
    description="Gender:",
    layout=widgets.Layout(width="300px"),
)
income_input = widgets.FloatText(
    description="Income:", value=50000, layout=widgets.Layout(width="300px")
)

previous_purchases_input = widgets.IntText(
    description="Previous Purchases:",
    value=5,
    layout=widgets.Layout(width="300px"),
    style={"description_width": "150px"},  
)

loyalty_points_input = widgets.IntText(
    description="Loyalty Points:",
    value=100,
    layout=widgets.Layout(width="300px"),
    style={"description_width": "150px"},  
)

# Dropdown for selecting the campaign 
campaign_dropdown = widgets.Dropdown(
    options=[(row["CampaignName"], index) for index, row in campaigns.iterrows()],
    description="Select Campaign:",
    layout=widgets.Layout(width="300px"),
    style={"description_width": "150px"},  
)

# Submit button
predict_button = widgets.Button(
    description="Predict", layout=widgets.Layout(width="200px")
)

# Display widgets
title = widgets.HTML(value="<h2>Customer Conversion Prediction Tool</h2>")
display(title)

display(
    age_input,
    gender_input,
    income_input,
    previous_purchases_input,
    loyalty_points_input,
    campaign_dropdown,
    predict_button,
)

out = widgets.Output()

display(out)



# This section is responsible for handling the prediction button click event
# It retrieves the selected campaign information, one-hot encodes the campaign channels and types,
# and prepares the user data for prediction.
def on_predict_button_clicked(b):
    # Retrieve selected campaign information
    selected_index = campaign_dropdown.value
    selected_campaign = campaigns.iloc[selected_index]

    # One-hot encode CampaignChannel and CampaignType
    campaign_channel = selected_campaign["CampaignChannel"]
    campaign_type = selected_campaign["CampaignType"]

    # Initialize one-hot encoding dictionaries
    campaign_channels = ["PPC", "SEO", "Social Media", "Referral"]
    campaign_types = ["Consideration", "Conversion", "Retention"]

    campaign_encoded = {}

    # One-hot encode campaign channels
    for channel in campaign_channels:
        key = "CampaignChannel_" + channel
        if campaign_channel == channel:
            campaign_encoded[key] = 1
        else:
            campaign_encoded[key] = 0

    # One-hot encode campaign types
    for type_ in campaign_types:
        key = "CampaignType_" + type_
        if campaign_type == type_:
            campaign_encoded[key] = 1
        else:
            campaign_encoded[key] = 0

    # Create user data dictionary
    user_data = {
        "Age": age_input.value,
        "Gender_Male": 1 if gender_input.value == "Male" else 0,
        "Income": income_input.value,
        **campaign_encoded,
        "AdSpend": selected_campaign["AdSpend"],
        "ClickThroughRate": selected_campaign["ClickThroughRate"],
        "ConversionRate": selected_campaign["ConversionRate"],
        "WebsiteVisits": selected_campaign["WebsiteVisits"],
        "PagesPerVisit": selected_campaign["PagesPerVisit"],
        "TimeOnSite": selected_campaign["TimeOnSite"],
        "SocialShares": selected_campaign["SocialShares"],
        "EmailOpens": selected_campaign["EmailOpens"],
        "EmailClicks": selected_campaign["EmailClicks"],
        "PreviousPurchases": previous_purchases_input.value,
        "LoyaltyPoints": loyalty_points_input.value,
    }

    # Convert user_data to DataFrame
    user_data_df = pd.DataFrame([user_data])

    # Ensure the columns match the training data
    user_data_df = user_data_df.reindex(columns=X.columns, fill_value=0)

    # Predict the probability of conversion for the user
    user_prob = rf_model.predict_proba(user_data_df)[:, 1]
    # Determine if the user is likely to convert based on the optimal threshold
    user_pred_optimal = (user_prob >= optimal_threshold).astype(int)

    with out:
        out.clear_output(wait=True)  # Clear any previous output
        if user_pred_optimal[0] == 1:
            print("The customer is likely to convert.")
        else:
            print("The customer is not likely to convert.")


predict_button.on_click(on_predict_button_clicked)

