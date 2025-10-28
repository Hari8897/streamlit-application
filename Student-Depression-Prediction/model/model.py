# import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle


def get_clean_data():
    # Load the data
    data = pd.read_csv(r"C:\Python projects\student depression data set\data\Depression Student Dataset.csv")
    return data

# Get unique vqlues in each categorical columns
def get_unique_values(data):
    data = get_clean_data()
    data_categorical = []
    for col in data.columns:
        if data[col].dtype == 'object':
            data_categorical.append(col)
            print(f"Unique values in {col}: {data[col].unique()}")
    return data_categorical

# Preprocess the data
def get_Preprocessing_data():
    data = get_clean_data()
    def safe_map(column, mapping, column_name):
        # safely map column values using a predefined mapping.
        try:
            return column.map(mapping)
        except Exception as e:
            print(f"Error mapping values in column '{column_name}'. Unexpected value detected. Please check the dataset.")
        return e
    try:
        # Handle Gender Mapping
        if 'Gender' in data.columns:
            gender_mapping = {'Male': 0, 'Female': 1}
            data['Gender'] = safe_map(data['Gender'], gender_mapping, 'Gender')
        
        # Handle Sleep Duration Mapping   
        if 'Sleep Duration' in data.columns:
            sleep_mapping = {
                'More than 8 hours':0,
                '7-8 hours' : 1, 
                '5-6 hours':2,
                'Less than 5 hours':3
            }
            data['Sleep Duration'] = safe_map(data['Sleep Duration'], sleep_mapping, 'Sleep Duration')
        
        # Handle Dietary Habits Mapping
        if 'Dietary Habits' in data.columns:
            dietary_mapping = {'Unhealthy':0, 'Moderate':1,
                               'Healthy':2}
            data['Dietary Habits'] = safe_map(data['Dietary Habits'], dietary_mapping, 'Dietary Habits')  

        # Handle Sucidal Thoughts Mapping
        if 'Have you ever had suicidal thoughts ?' in data.columns:
            suicidial_mapping = {'Yes':1, "No":0}
            data['Have you ever had suicidal thoughts ?']=safe_map(data['Have you ever had suicidal thoughts ?'],
                                                                 suicidial_mapping,
                                                                 'Have you ever had suicidal thoughts ?')
        # Handle Family History of Mental Illness Mapping
        if 'Family History of Mental Illness' in data.columns:
            family_mapping = {'Yes':1, "No":0}
            data['Family History of Mental Illness']=safe_map(data['Family History of Mental Illness'],
                                                            family_mapping,
                                                            'Family History of Mental Illness')
        # Handle Depression Mapping
        if 'Depression' in data.columns:
            depression_mapping = {'Yes': 1, 'No': 0}
            data['Depression'] = safe_map(data['Depression'], depression_mapping, 'Depression')
    except KeyError as e:
        print(f"KeyError: Missing column '{e}' in the dataset. Please ensure the dataset contains the required columns.")
        
    return data
# Get the correlation matrix
def get_correlation_matrix(data):
    data = get_Preprocessing_data()
    correlation_matrix = data.corr()
    return correlation_matrix

# Model training
def train_model(data):
     
    data = get_Preprocessing_data()
    X = data.drop(['Depression'], axis=1)
    y = data['Depression']
    
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
        
    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=42)
    # fit the data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # make predictions
    y_pred = model.predict(X_test)
    # evaluate the model with Accuracy
    print('================= ** Model Evoluation ** =================')
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    return model,scaler
    
# print the result
def main():
    # get clean data 
    data = get_clean_data()
    # get preprocessing data
    # print(data.head(5))
    print("======================**After Preprocessing**===================")
    preprocessing_data = get_Preprocessing_data()
    print(preprocessing_data.head(5))
    print('======================** Unique categorical columns**==================')
    print(get_unique_values(data))
    print('======================**Correlation Matrix**==================')
    print(get_correlation_matrix(data))
    
    # train the model
    model, scaler = train_model(data)
    
    # # save the model
    filename = 'model/finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    # # save the scaler
    filename = 'model/scaler.sav'
    pickle.dump(scaler, open(filename, 'wb'))
    

if __name__ == '__main__':
    main()