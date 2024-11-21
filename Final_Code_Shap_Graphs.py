
#Importing all the library and packages require to run code

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import LocalOutlierFactor
import shap
import fasttreeshap
from sklearn.feature_selection import SelectKBest, chi2
from lime import lime_tabular
from scipy.stats import norm
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# create Dataframe by reading csv file 
df = pd.read_csv("C:\\Users\\pruch\\OneDrive\\Desktop\\Final_Code_Shap_Graphs\\diabetic_data.csv")
  
#converting age range into integer by calculating average
def age_mapping():
    return {
        '[0-10)': 5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95
        }

#Admission Type Id mapping as per data given in UCI machine leaning link
def mapped_admission_type_id():
    return {
        1 : 'Emergency',
        2 : 'Urgent',
        3 :	'Elective',
        4 : 'Newborn',
        5 :	 np.nan,
        6 :	 np.nan,
        7 : 'Trauma Center',
        8 :	 np.nan
    }

#Discharge Type Id mapping as per data given in UCI machine leaning link
def mapped_discharge():
    return {
        1:"Discharged to Home",
        6:"Discharged to Home",
        8:"Discharged to Home",
        13:"Discharged to Home",
        19:"Discharged to Home",
        18:np.nan,25:np.nan,26:np.nan,
        2:"Other",3:"Other",4:"Other",
        5:"Other",7:"Other",9:"Other",
        10:"Other",11:"Other",12:"Other",
        14:"Other",15:"Other",16:"Other",
        17:"Other",20:"Other",21:"Other",
        22:"Other",23:"Other",24:"Other",
        27:"Other",28:"Other",29:"Other",30:"Other"
        }

#Admission Source Id mapping as per data given in UCI machine leaning link
def mapped_adm_source():
    return {
        1:"Referral",2:"Referral",3:"Referral",
        4:"Other",5:"Other",6:"Other",10:"Other",22:"Other",25:"Other",
        9:"Other",8:"Other",14:"Other",13:"Other",11:"Other",
        15:np.nan,17:np.nan,20:np.nan,21:np.nan,7:"Emergency"
        }
    
def mapped_max_glu_serum(): 
    return {
            ">200": 1, ">300": 1,"Norm": 0, np.nan: 1
        }
    
def mapped_A1Cresult(): 
    return {
            ">7": 1, ">8": 1, "Norm": 0, np.nan: 1 
        }

def mapped_medication():
    return{
        "No": 0, "Steady": 0, "Up": 1, "Down": 1
        }

def boxplot(col_name):
    plt.figure(figsize=(10, 6))
    plt.boxplot(df[col_name])
    plt.title(col_name)
    plt.ylabel('Days')
    plt.show()

def mapped_diag(value):
    # Mapping logic
    if str(value).startswith("250"):
        return "Diabetes"
    elif str(value).startswith("V") or str(value).startswith("E") or str(value).startswith("365.44"):
        return "Other diseases"
    elif 140 <= int(value) < 240:
        return "Neoplasms"
    elif (390 <= int(value) < 460) or (int(value) == 785):
        return "Circulatory"
    elif (460 <= int(value) < 520) or (int(value) == 786):
        return "Respiratory"
    elif (520 <= int(value) < 580) or (int(value) == 787):
        return "Digestive"
    elif 800 <= int(value) < 1000:
        return "Injury"
    elif 700 <= int(value) < 740:
        return "Musculoskeletal"
    elif (500 <= int(value) < 630) or (int(value) == 788):
        return "Genitourinary"
    else:
        return "Other diseases"

def medicine_change_count(col_name, df):
    # Replace medicine change with their values
    map_metformin = mapped_medication()
    df[col_name] = df[col_name].map(map_metformin)

    #unique value of medicine
    print("${%dcol_name}: ", df[col_name].unique())

    # Convert medicine values to integers
    df[col_name] = df[col_name].astype(int)
    
    return df

def boxplot_for_outliers_lof(df, columns):
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20, 15))
    fig.subplots_adjust(hspace=0.5)
    count = 0
    
    for i in range(4):
        for j in range(4):
            if count < len(columns):
                sns.boxplot(x=df[columns[count]], ax=ax[i][j], palette="Wistia")
                ax[i][j].set_title(f"Boxplot of {columns[count]}")
                count += 1
            else:
                ax[i][j].axis('off')  # Hide empty subplots
    plt.show()
    
def random_forest(X_train, X_test, Y_train, Y_test):
    # Train Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, Y_train)

    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Identify misclassified instances
    misclassified_indices = np.where(y_pred != Y_test)[0]
    print(f'Number of misclassified instances: {len(misclassified_indices)}')

    # Calculate evaluation metrics
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    y_prob = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(Y_test, y_prob)

    print(f'RF Accuracy: {accuracy:.2f}')
    print(f'RF Precision: {precision:.2f}')
    print(f'RF Recall: {recall:.2f}')
    print(f'RF F1 Score: {f1:.2f}')
    print(f'RF AUC: {auc:.2f}')

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Not Readmitted (0)', 'Readmitted (1)'],
        mode='classification'
    )

    # Explain misclassified instances using LIME
    for index in misclassified_indices[:3]:  # Limit to first 5 misclassifications
        instance = X_test.iloc[index]
        true_label = Y_test.iloc[index]
        predicted_label = y_pred[index]
        exp = explainer.explain_instance(instance, rf.predict_proba)
        
        print(f'\nExplanation for misclassified instance {index}:')
        print(f'True label: {true_label}, Predicted label: {predicted_label}')
        print("Top features contributing to the prediction:")
        for feature, value in exp.as_list():
            print(f"{feature}: {value}")
            
            # Visualize the explanation
        # Visualize the explanation
    fig = plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title(f'LIME Explanation for Instance {index}')
    plt.tight_layout()
    plt.savefig(f'lime_explanation_instance_{index}.png')
    plt.show()

    return rf, y_pred
    
def xgboost(X_train, X_test, Y_train, Y_test):
    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
   
    # Calculate evaluation metrics
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
   
    # Calculate AUC
    y_prob = model.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for the positive class
    auc = roc_auc_score(Y_test, y_prob)

    # Print the results
    print(f'XGBOOST Accuracy: {accuracy:.2f}')
    print(f'XGBOOST Precision: {precision:.2f}')
    print(f'XGBOOST Recall: {recall:.2f}')
    print(f'XGBOOST F1 Score: {f1:.2f}')
    print(f'XGBOOST AUC: {auc:.2f}')
   
def select_feature(X_resampled, Y_resampled ):
    k = 19  # Number of features to select
    chi2_selector = SelectKBest(chi2, k=k)
    X_kbest = chi2_selector.fit_transform(X_resampled, Y_resampled)

    # Get the selected feature names
    selected_features = X_resampled.columns[chi2_selector.get_support()]

    print(f"Selected features using K Best (k={k}):", selected_features.tolist())

    print(selected_features.shape)

    # Keep only the selected features in a new DataFrame
    X_selected = X_resampled[selected_features]
    
    return X_selected

def standard_scale(X_resampled):
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    # Print the shape of the scaled dataset
    print("Shape of scaled resampled training set:", X_resampled_scaled.shape)
    
    return X_resampled_scaled
    
####################################### Preprocessing #################################

#replace ? as null
df = df.replace('?', np.nan)

print(df.isin(['?']).any())

#check shap of the dataframe
print(df.shape)

#converting target variable to binary
df["readmitted"] = df["readmitted"].replace({"NO":0, "<30":1, ">30":0})

if(df["readmitted"]==1).any():
    pass
else:
    # Drop duplicates, keeping the first occurrence
    df = df.drop_duplicates(subset='patient_nbr', keep='first')

#dropping attirubute with higher null values
df.drop(['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr'],axis=1,inplace=True)

df.info()
print(df.shape)

#check null values in percentage
print(df.isnull().sum() * 100 / len(df))

# Remove rows where gender is 'Unknown/Invalid' and reset the index in one line
df = df[df['gender'] != 'Unknown/Invalid'].reset_index(drop=True)

#convert age variable to integer
# Replace age groups with their average values
df['age'] = df['age'].astype(str)

# Apply the mapping
age_map = age_mapping()
df['age'] = df['age'].map(age_map).astype(int)

#unique value of age
print(df['age'].unique())

df.head()

# Apply the mapping admission type
map_adm_type = mapped_admission_type_id()
df['admission_type_id'] = df['admission_type_id'].map(map_adm_type)

#unique value of admission type
print(df['admission_type_id'].unique())

# Apply the mapping to discharge 
map_discharge = mapped_discharge()
df['discharge_disposition_id'] = df['discharge_disposition_id'].map(map_discharge)

#unique value of admission type
print(df['discharge_disposition_id'].unique())

# Apply the mapping to admission source ID 
adm_map = mapped_adm_source()
df['admission_source_id'] = df['admission_source_id'].map(adm_map)

#unique value of admission source ID
print(df['admission_source_id'].unique())

# Apply the mapping to max_glu_serum
map_max_glu = mapped_max_glu_serum()
df['max_glu_serum'] = df['max_glu_serum'].map(map_max_glu)

#unique value of max_glu_serum
print("max glue: ", df['max_glu_serum'].unique())

# Replace A1Cresult with their values
map_A1Cresult = mapped_A1Cresult()
df['A1Cresult'] = df['A1Cresult'].map(map_A1Cresult)

#unique value of A1Cresult
print("A1Cresult: ", df['A1Cresult'].unique())

# Convert A1Cresult values to integers
df['A1Cresult'] = df['A1Cresult'].astype(int)

# Convert max_glu_serum values to integers
df['max_glu_serum'] = df['max_glu_serum'].astype(int)

#drop all NA value
df = df.dropna()
df.info()

# Replace diag code with their values
# Apply the mapping function to the 'diag_1' column
df['diag_1'] = df['diag_1'].apply(mapped_diag)

# Apply the mapping function to the 'diag_2' column
df['diag_2'] = df['diag_2'].apply(mapped_diag)

# Apply the mapping function to the 'diag_3' column
df['diag_3'] = df['diag_3'].apply(mapped_diag)

# Unique values of diag code
print("Unique values in diag_1: ", df['diag_1'].unique())
print("Unique values in diag_2: ", df['diag_2'].unique())
print("Unique values in diag_3: ", df['diag_3'].unique())

#creating service_utilization column by summing up number_outpatient, number_emergency, number_inpatient

# Create service_utilization column by summing the specified columns
df['service_utilization'] = df[['number_outpatient', 'number_emergency', 'number_inpatient']].sum(axis=1)

# Drop the original columns (without using inplace=True)
df.drop(['number_outpatient', 'number_emergency', 'number_inpatient'], axis=1, inplace=True)

# Display updated DataFrame
print("\nUpdated DataFrame:")
print(df)
print("\nColumns in the updated DataFrame:", df.columns.tolist())

#Summing up 23 variables and creating new column "medication_change_count" (metformin	repaglinide	nateglinide	chlorpropamide	glimepiride	acetohexamide	glipizide	glyburide	tolbutamide	pioglitazone	rosiglitazone	acarbose	miglitol	troglitazone	tolazamide	examide	citoglipton	insulin	glyburide-metformin	glipizide-metformin	glimepiride-pioglitazone	metformin-rosiglitazone	metformin-pioglitazone)
# as they give information about specific medication changed or not. 

df = medicine_change_count("metformin", df)
df = medicine_change_count("repaglinide", df)
df = medicine_change_count("nateglinide", df)
df = medicine_change_count("chlorpropamide", df)
df = medicine_change_count("glimepiride", df)
df = medicine_change_count("acetohexamide", df)
df = medicine_change_count("glipizide", df)
df = medicine_change_count("glyburide", df)
df = medicine_change_count("tolbutamide", df)
df = medicine_change_count("pioglitazone", df)
df = medicine_change_count("rosiglitazone", df)
df = medicine_change_count("acarbose", df)
df = medicine_change_count("miglitol", df)
df = medicine_change_count("troglitazone", df)
df = medicine_change_count("tolazamide", df)
df = medicine_change_count("examide", df)
df = medicine_change_count("citoglipton", df)
df = medicine_change_count("insulin", df)
df = medicine_change_count("glyburide-metformin", df)
df = medicine_change_count("glipizide-metformin", df)
df = medicine_change_count("glimepiride-pioglitazone", df)
df = medicine_change_count("metformin-rosiglitazone", df)
df = medicine_change_count("metformin-pioglitazone", df)

#adding new column:
# Create medication change count column by summing the specified columns
df['medication_change_count'] = df[["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]].sum(axis=1)

# Drop the original columns (without using inplace=True)
df.drop(["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"], axis=1, inplace=True)

# Display updated DataFrame
print("\nUpdated DataFrame:")
print(df)
print("\nColumns in the updated DataFrame:", df.columns.tolist())

#convert gender, change and diabetesMed into binary
df['change'] = df['change'].replace('Ch', 1)
df['change'] = df['change'].replace('No', 0)

df['gender'] = df['gender'].replace('Male', 1)
df['gender'] = df['gender'].replace('Female', 0)

df['diabetesMed'] = df['diabetesMed'].replace('Yes', 1)
df['diabetesMed'] = df['diabetesMed'].replace('No', 0)

df['discharge_disposition_id'] = df['discharge_disposition_id'].replace('Other', 1)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace('Discharged to Home', 0)

#df.to_csv('file_name_categorical.csv', index=False)

#implement label encoder for race, admission_type_id, admission_source_id, diag_1, diag_2, diag_3

lb = LabelEncoder()

# List of columns to encode
columns_to_encode = ['race', 'admission_type_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3']

# Apply LabelEncoder to each specified column
for column in columns_to_encode:
    df[column] = lb.fit_transform(df[column])

# Display updated DataFrame
print("\nUpdated DataFrame with Encoded Variables:")

#df.to_csv('file_name_finalEncoder.csv', index=False)

# To get numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Print numerical column names
print("Numerical Columns in the DataFrame: ", numerical_columns)


# Define numerical columns
numerical_columns = ['age', 'time_in_hospital', 'num_lab_procedures', 
                     'num_procedures', 'num_medications', 'number_diagnoses', 
                     'service_utilization', 'medication_change_count']

# Initialize the LocalOutlierFactor model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

outlier_predictions = lof.fit_predict(df[numerical_columns])

# Create a new DataFrame without outliers
df_no_outliers = df[outlier_predictions != -1]

# Print the number of outliers removed
print(f"Original dataset size: {df.shape[0]}")
print(f"New dataset size after removing outliers: {df_no_outliers.shape[0]}")

# Call the function to display boxplots for numerical columns
boxplot_for_outliers_lof(df, numerical_columns)

# Remove outliers (LOF assigns -1 to outliers)
df_no_outliers = df[outlier_predictions != -1]

# Optionally check the number of outliers removed
print(f"Original dataset size: {df.shape[0]}")
print(f"New dataset size after removing outliers: {df_no_outliers.shape[0]}")

print("\nColumns in the updated DataFrame:", df_no_outliers.columns.tolist())

#df_no_outliers.to_csv('C:\\Users\\Dhava\\OneDrive\\Documents\\Nikita Project\\Code\\final_no_outliers.csv', index=False)

#Plot he readmission column
readmission_counts = df['readmitted'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x=readmission_counts.index.astype(str), y=readmission_counts.values, palette='viridis')

# Add labels and title
plt.xlabel('Readmission Status')
plt.ylabel('Count')
plt.title('Distribution of Readmission Status')
plt.xticks([0, 1], ['No Readmission (0)', 'Readmission (1)'])  # Custom labels for clarity

# Show the plot
plt.show()

print(df_no_outliers.shape)

#saving data for testing model:
# Randomly select 50 rows
test_data = df_no_outliers.sample(n=50, random_state=42)

# Remove the selected rows from the original DataFrame
df_no_outliers = df_no_outliers.drop(test_data.index)

# Save the test data to a CSV file
test_data.to_csv('C:\\Users\\pruch\\OneDrive\\Desktop\\Final_Code_Shap_Graphs\\test_data.csv', index=False)

print(df_no_outliers.shape)

# Assuming 'df' is your DataFrame
X = df_no_outliers.drop(columns=['readmitted'])
Y = df_no_outliers['readmitted']

# Apply SMOTE to the training set only
smote = SMOTE(random_state=42)

# Fit and resample the training data
X_resampled, Y_resampled = smote.fit_resample(X, Y)

# Check the balance of the classes after SMOTE
print("Original dataset size:", Y.value_counts())
print("Resampled dataset size:", Y_resampled.value_counts())

print(X_resampled.shape)

print("\nColumns in the updated DataFrame:", X_resampled.columns.tolist())

############# Feature selection using K Best

X_selected = select_feature(X_resampled, Y_resampled)
print("\nColumns in the updated DataFrame:", X_selected.columns.tolist())

### feature scaling using standard scaler
scaler = StandardScaler()
X_selected_scaled = scaler.fit_transform(X_selected)

feature_names = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'A1Cresult', 'change', 'diabetesMed', 'service_utilization', 'medication_change_count']  # Replace with your actual feature names
X_scaled_df = pd.DataFrame(X_selected_scaled, columns=feature_names)

# Now you can split the resampled data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled_df, Y_resampled, test_size=0.3, shuffle=True, stratify=Y_resampled, random_state=42)

# Fit Random Forest model
model, Y_pred = random_forest(X_train, X_test, Y_train, Y_test)

####################################################################

# Create the FastTreeExplainer
explainer = fasttreeshap.TreeExplainer(model)

### misclassified Instance
# Step 1: Identify misclassified instances
misclassified_indices = (Y_pred != Y_test)
misclassified_X = X_test[misclassified_indices]  # Select only the first 500 misclassified instances
misclassified_Y_true = Y_pred[misclassified_indices]

print(misclassified_Y_true.shape)

# Step 2: Create a DataFrame for misclassified instances
misclassified_df = pd.DataFrame(misclassified_X, columns=X_scaled_df.columns)
#misclassified_df['True_Label'] = misclassified_Y_true

print("Misclassified dataframe shape:", misclassified_df.columns)

######################################################################

# Calculate SHAP values for the new data point
shap_values = explainer.shap_values(misclassified_df)

print("Shape of shap_values:", np.array(shap_values).shape)

# Extract SHAP values for the positive class (assuming index 1 is the positive class)
shap_values_array = np.array(shap_values[1])  # Use index 1 for the positive class

# Create a dataframe with SHAP values
shap_df = pd.DataFrame(shap_values_array, columns=X_test.columns)

# Add a prefix to column names to indicate they are SHAP values
shap_df.columns = ['SHAP_' + col for col in shap_df.columns]

# Add index from original data
shap_df.index = misclassified_X.index

# Display the resulting dataframe
print(shap_df.columns)

# Step 4: Merge SHAP values with the misclassified DataFrame
combined_df = pd.concat([misclassified_df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

combined_df["readmitted"] = misclassified_Y_true.values

# Step 5: Save to CSV (optional)
combined_df.to_csv('misclassified_with_shap.csv', index=False)

print("Combined DataFrame with SHAP values saved.")

######################################################################

# Example new data
new_data = {
'race': 2,
'gender': 1,
'age': 95,
'admission_type_id': 0,
'discharge_disposition_id': 0,
'admission_source_id': 2,
'time_in_hospital': 4,
'num_lab_procedures': 42,
'num_procedures': 0,
'num_medications': 15,
'diag_1': 0,
'diag_2': 0,
'diag_3': 0,
'number_diagnoses': 9,
'A1Cresult': 1,
'change': 0,
'diabetesMed': 0,
'service_utilization': 0,
'medication_change_count': 0
}

# Step 4: Create a DataFrame for the new data
new_data_df = pd.DataFrame([new_data])  # Wrap the dictionary in a list

# Step 5: Scale the new data
new_data_scaled = scaler.transform(new_data_df)

# Assuming new_data_scaled is a numpy array
new_data_scaled_df = pd.DataFrame(new_data_scaled, columns=new_data_df.columns)

# Print the DataFrame to verify
print(new_data_scaled_df)

# Step 6: Make predictions
y_pred_newdata = model.predict(new_data_scaled)
y_pred_newdata_prob = model.predict_proba(new_data_scaled)

# Calculate SHAP values for the new data point
shap_values_newdata = explainer.shap_values(new_data_scaled)

print(shap_values_newdata)

print("Shape of shap_values:", np.array(shap_values_newdata).shape)

# Extract SHAP values for the positive class (assuming index 1 is the positive class)
shap_values_array_newdata = np.array(shap_values_newdata[1])  # Use index 1 for the positive class

# Create a dataframe with SHAP values
shap_df_newdata = pd.DataFrame(shap_values_array_newdata, columns=X_test.columns)

# Add a prefix to column names to indicate they are SHAP values
shap_df_newdata.columns = ['SHAP_' + col for col in X_test.columns]

# Step 4: Merge SHAP values with the misclassified DataFrame
combined_df_newdata = pd.concat([new_data_scaled_df.reset_index(drop=True), shap_df_newdata.reset_index(drop=True)], axis=1)

#combined_df = pd.read_csv("C:\\Users\\pruch\\OneDrive\\Desktop\\Final_Code_Shap_Graphs\\misclassified_with_shap.csv")

x_shap = combined_df.drop(columns=['readmitted'])
y_shap = combined_df['readmitted']

# Step 6: Create and fit KNN classifier
k = 5  # You can adjust this value
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_shap, y_shap)
                        
# Step 7: Find k nearest neighbors based on SHAP values
distances, indices = knn.kneighbors(combined_df_newdata)

# Step 8: Get the actual outcomes of the nearest neighbors
nearest_neighbors_outcomes = y_shap.iloc[indices[0]]

# Step 9: Make a prediction based on the majority vote of nearest neighbors
knn_prediction = nearest_neighbors_outcomes.mode()[0]

# Print the prediction
print("RF prediction:", y_pred_newdata)
print("RF prediction probability:", y_pred_newdata_prob)

# Step 10: Print results
print(f"KNN Prediction based on SHAP values: {knn_prediction}")
print(f"Nearest neighbors outcomes: {nearest_neighbors_outcomes.tolist()}")
# Print distances of the nearest neighbors
print(f"Distances of nearest neighbors: {distances[0].tolist()}")

############################# feature importance

def shap_force_plot():
    # Convert SHAP values to a numpy array
    shap_values = shap_df_newdata.values

    # Get feature names
    feature_names = shap_df_newdata.columns.tolist()

    # Get the base value (expected value)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]  # Assuming we're interested in class 1 (adjust if needed)

    # Select top 5-6 features by absolute SHAP values
    top_n = 6  # Change this to 5 if you want exactly 5 features
    shap_values_instance = shap_values[0]
    top_features_idx = np.argsort(np.abs(shap_values_instance))[-top_n:]  # Get indices of top features

    # Filter SHAP values, feature names, and data to include only the top features
    top_shap_values = shap_values_instance[top_features_idx]
    top_feature_names = [feature_names[i] for i in top_features_idx]
    top_data = new_data_scaled_df.values[0][top_features_idx]

    # Create an Explanation object with only the top features
    explanation = shap.Explanation(
    values=top_shap_values,
    base_values=base_value,
    data=top_data,
    feature_names=top_feature_names
    )

    # Create the force plot
    shap.plots.force(explanation, matplotlib=True)
    plt.title("SHAP Force Plot - Top Features")
    plt.tight_layout()
    plt.show()

def summary_plot():
    # Calculate the mean absolute SHAP values for each feature across the dataset
    mean_abs_shap_values = np.abs(shap_values_newdata[1]).mean(axis=0)

    # Get the indices of the top 6 features by mean SHAP value
    top_features_idx = np.argsort(mean_abs_shap_values)[-6:]

    # Filter SHAP values and feature names for only the top 6 features
    top_shap_values = shap_values_newdata[1][:, top_features_idx]
    top_feature_names = [feature_names[i] for i in top_features_idx]
    top_data = new_data_scaled[:, top_features_idx]

    # Create the bar plot for the top 6 features
    shap.summary_plot(top_shap_values, top_data, feature_names=top_feature_names, plot_type="bar")

def shap_waterfall_plot():
    # Convert SHAP values to a numpy array
    shap_values = shap_df_newdata.values[0]  # SHAP values for a single instance

    # Get feature names
    feature_names = shap_df_newdata.columns.tolist()

    # Get the base value (expected value)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]  # Assuming we're interested in class 1 (adjust if needed)

    # Create an Explanation object for the waterfall plot
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=new_data_scaled_df.values[0],  # Feature values for the instance
        feature_names=feature_names
    )

    # Plot the waterfall plot
    shap.plots.waterfall(explanation)
    plt.title("SHAP Waterfall Plot for Single Prediction")
    plt.tight_layout()
    plt.show()
    
################################################## validation of prediction

rf_prediction = y_pred_newdata
prob_confidence = y_pred_newdata_prob[0]
objections = []

if rf_prediction[0] != knn_prediction:
    if rf_prediction[0] == 0:
        if prob_confidence[0] > 0.70:  # 70% confidence for class 0
            if any(distance < 2.0 for distance in distances[0]):
                objections.append("OBJECTION Raised")
                shap_force_plot()
                summary_plot()
                
            else:
                objections.append("NO OBJECTION Raised")
        else:
                objections.append("OBJECTION Raised")
                shap_force_plot()
                summary_plot()

    elif rf_prediction[0] == 1:
        if prob_confidence[0] > 0.70:  # 70% confidence for class 0
            if any(distance < 2.0 for distance in distances[0]):
                objections.append("OBJECTION Raised")
                shap_force_plot()
                summary_plot()

            else:
                objections.append("NO OBJECTION Raised")
        else:
                objections.append("OBJECTION Raised")
                shap_force_plot()
                summary_plot()

if rf_prediction[0] == knn_prediction:
    if rf_prediction[0] == 0:
        if prob_confidence[0] > 0.70:  # 70% confidence for class 0
            if any(distance < 2.0 for distance in distances[0]):
                objections.append("OBJECTION Raised")
            else:
                objections.append("NO OBJECTION Raised")
                shap_force_plot()
                summary_plot()
        else:
            objections.append("OBJECTION Raised")

    elif rf_prediction[0] == 1:
        if prob_confidence[0] > 0.70:  # 70% confidence for class 0
            if any(distance < 2.0 for distance in distances[0]):
                objections.append("OBJECTION Raised")
            else:
                objections.append("NO OBJECTION Raised")
                shap_force_plot()
                summary_plot()
        else:
            objections.append("OBJECTION Raised")

                
# Print the objections
if objections:
    for objection in objections:
        print(objection)
else:
    print("No objections raised.")
        
###############################################################
"""
#Code to calculate distance measure
# Assuming misclassified_X and X_train are already defined

# Calculate distances using NearestNeighbors
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_train)
distances, _ = nn.kneighbors(misclassified_X)

# Flatten the distances array
distances_flat = distances.flatten()

# Calculate critical threshold
critical_threshold = distances_flat.max()
print(f"Critical threshold: {critical_threshold}")

# Calculate some statistics
mean_distance = np.mean(distances_flat)
median_distance = np.median(distances_flat)
std_distance = np.std(distances_flat)

# Create a histogram of the distances
plt.figure(figsize=(12, 6))
sns.histplot(distances_flat, kde=True, bins=50)
plt.title("Distribution of Distances to Nearest Misclassified Point")
plt.xlabel("Distance")
plt.ylabel("Frequency")

# Add vertical lines for mean, median, and critical threshold
plt.axvline(mean_distance, color='r', linestyle='--', label=f'Mean: {mean_distance:.2f}')
plt.axvline(median_distance, color='g', linestyle='--', label=f'Median: {median_distance:.2f}')
plt.axvline(critical_threshold, color='b', linestyle='--', label=f'Critical Threshold: {critical_threshold:.2f}')

plt.legend()
plt.show()

# Print additional statistics
print(f"Mean distance: {mean_distance}")
print(f"Median distance: {median_distance}")
print(f"Standard deviation of distances: {std_distance}")

# Calculate percentiles
percentiles = [25, 60, 65, 90, 95, 99]
for p in percentiles:
    print(f"{p}th percentile: {np.percentile(distances_flat, p):.2f}")

# Count points within certain thresholds
thresholds = [2, 2.5, 3]
for t in thresholds:
    count = np.sum(distances_flat < t)
    percentage = (count / len(distances_flat)) * 100
    print(f"Points within distance {t}: {count} ({percentage:.2f}%)")

# Step 1: Train Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)

# Step 2: Generate Prediction Probabilities
y_prob = rf.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Given threshold is 70%
threshold_70 = 0.70
y_pred_70 = (y_prob >= threshold_70).astype(int)

# Calculate Precision and Recall at 70% threshold
precision_70 = precision_score(Y_test, y_pred_70)
recall_70 = recall_score(Y_test, y_pred_70)
print(f'Precision at 70% threshold: {precision_70:.2f}')
print(f'Recall at 70% threshold: {recall_70:.2f}')

# Calculate ROC and AUC at threshold 70%
fpr_70, tpr_70, thresholds_70 = roc_curve(Y_test, y_prob)
auc_70 = roc_auc_score(Y_test, y_prob)
print(f'AUC at 70% threshold: {auc_70:.2f}')

# Optional: Visualize ROC curve and AUC
plt.figure(figsize=(8, 6))
plt.plot(fpr_70, tpr_70, color='blue', label=f'ROC curve (AUC = {auc_70:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.title('ROC Curve at 70% Threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


#####################################################

#calculation of prediction probability threshold
# Step 1: Train Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)

# Step 2: Generate Prediction Probabilities
y_prob = rf.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Given threshold is 70%
threshold_70 = 0.70
y_pred_70 = (y_prob >= threshold_70).astype(int)

# Calculate Precision and Recall at 70% threshold
precision_70 = precision_score(Y_test, y_pred_70)
recall_70 = recall_score(Y_test, y_pred_70)
print(f'Precision at 70% threshold: {precision_70:.2f}')
print(f'Recall at 70% threshold: {recall_70:.2f}')

# Calculate ROC and AUC at threshold 70%
fpr_70, tpr_70, thresholds_70 = roc_curve(Y_test, y_prob)
auc_70 = roc_auc_score(Y_test, y_prob)
print(f'AUC at 70% threshold: {auc_70:.2f}')

# Optional: Visualize ROC curve and AUC
plt.figure(figsize=(8, 6))
plt.plot(fpr_70, tpr_70, color='blue', label=f'ROC curve (AUC = {auc_70:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.title('ROC Curve at 70% Threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
"""