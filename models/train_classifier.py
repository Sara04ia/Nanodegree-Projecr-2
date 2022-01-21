import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn import multioutput
from sklearn.multioutput import MultiOutputClassifier
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import re
import pickle
import os
from scipy.stats import gmean


def load_data(database_filepath):
    # load data from database
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Disa_Repose_DB', con=engine)
    
    X = df ['message']
    Y = df.drop(['id','message','original', 'genre'], axis = 1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words("english")
    words_tokenize = word_tokenize (text)
    Stemmer = [PorterStemmer().stem(word) for word in words_tokenize]
    words_lemmatize = [WordNetLemmatizer().lemmatize(word) for word in Stemmer if word not in stop_words]
    return words_lemmatize


def build_model():
    model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = { 'vect__max_df': (0.75, 1.0),
                'clf__estimator__n_estimators': [10, 20],
                'clf__estimator__min_samples_split': [2, 5]
              }
    cv = GridSearchCV(model,parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_predict_CV = model.predict (X_test)
    Y_predict_dataframe_cv = pd.DataFrame(Y_predict_CV, columns = Y_test.columns)
    for column in Y_test.columns:
         print('Column Name Is : {} \n'.format(column))
         print(classification_report(Y_test[column], Y_predict_dataframe_cv[column]))


def save_model(model, model_filepath):
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