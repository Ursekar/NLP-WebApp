# import libraries
import sqlalchemy as sqlal
import pandas as pd
import sys

import nltk as nk
from nltk.corpus import stopwords
nk.download(['punkt', 'wordnet'])
nk.download('stopwords')

import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

def load_data(database_filepath):
    # load data from database
    engine = sqlal.create_engine('sqlite:///' + str(database_filepath))
    print('\n The table names are ',engine.table_names())
    df = pd.read_sql('select*from disaster_table', con=engine)
    print('View Your dataframe \n',df.head())

    # Copying the dataframe for analysis and dropping unwanted columns 
    df_copy = df.drop(df.columns[[0,2]], axis=1)

    # Dropping null rows and checking shape
    df_copy.dropna(inplace = True)
    print('\n Shape of dataframe: \n', df_copy.shape)

    # Checking for any nulls left
    print('\n checking for nulls: \n', df_copy.isnull().sum())

    # Seperating inputs and outputs
    X = df_copy.iloc[:,0]
    Y = df_copy.iloc[:,2:]
    
    # The related seems to have three categories, hence replacing 2s with 1s for simplicity
    Y.related = Y.related.replace(2,1)
    print('Verifying if related has only 2 categories: ', Y.related.unique())
   
    # Fetching names of categories
    category_names = df_copy.columns[2:]
    
    return X,Y,category_names


def tokenize(text):
    
    stop_words = stopwords.words('english')
    clean_tokens = []
    for sent in text:
        sent = re.sub(r"[^a-zA -Z]", " ", sent.lower())
        tokens = nk.word_tokenize(sent)
        lemmatizer = nk.WordNetLemmatizer()
    
        for tok in tokens:
            if tok not in stop_words:
                clean_tok = lemmatizer.lemmatize(tok).strip()
                clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    
    # Building a pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    
    parameters = {'clf__estimator__n_estimators': [80, 120],
                  'clf__estimator__min_samples_split': [4, 6, 8],
                  #'clf__estimator__criterion': ['entropy', 'gini']
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    # Generate predictions
    y_pred = model.predict(X_test)
      
    # Getting classification report by each category
    print('\n The Classification report is as below: \n')
    print(classification_report(Y_test, y_pred, target_names=category_names))
    
def save_model(model, model_filepath):
    import pickle
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()