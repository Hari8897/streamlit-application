# import libraries
import pandas as pd
import pickle


# getting clean data

def get_clean_data():
    # Load the data
    data = pd.read_csv(r"C:\Python projects\Streamlit\Breast-cancer-prediction\data\data.csv")
    # drop the unwanted columns
    data=data.drop(['Unnamed: 32','id'], axis=1)
    # map the diagnosis column
    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    # return the clean data    
    return data


def create_model(data):
    # Import the necessary libraries
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # scale the data 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=42)
    # train the data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # preditct and test the model
    y_pred = model.predict(X_test)
    # print the accuracy and classification report
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model,scaler 

def main():
    data = get_clean_data()
    model, scaler = create_model(data)   
    # export the model using pickle
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()