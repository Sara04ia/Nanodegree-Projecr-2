# Nanodegree-Project2
## Disaster Response Pipelines

### Project Summary

This Project is part of Data Science Nanodegree Program and it will include a web app where the worker can input a new message and get classification results in several categories also The web app will display the visualizations of data.

This project is divided in the following key sections:
1) Processing data by building an ETL pipeline and it's contains :
	1. Loads the messages and categories dataset
	2. Merges the two datasets
	3. Cleans the data
	4. Stores it in a SQLite database
		![Screenshot (989)](https://user-images.githubusercontent.com/94194880/150601267-70dc3201-d8d3-42f4-bac5-17799a4ec725.png)

3) Build a machine learning pipeline to train and it's contains :
	1. Loads data from the SQLite database
	2. Splits data into training set and testing set
	3. Builds a text processing and machine learning pipeline
	4.Trains and tunes a model using GridSearchCV
	5. Outputs result on the test set
	6. Exports the final model as a pickle file
		![Screenshot (990)](https://user-images.githubusercontent.com/94194880/150601345-cac56f8c-fc46-4462-a12c-31ec969e041e.png)

5) Run a web app which can show model results in real time

Go to http://0.0.0.0:3001/

https://view6914b2f4-3001.udacity-student-workspaces.com/

![Screenshot (993)](https://user-images.githubusercontent.com/94194880/150691537-ccbcf950-456c-4161-8baa-cf752daacdb7.png)

### Executing Program:
First Run the following commands in the project's root directory to set up your database and model.
- Run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- Run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Second Run the following command in the app's directory to run your web app. python run.py



## Web application screenshots
![newplot](https://user-images.githubusercontent.com/94194880/150691573-2f2299a8-0d05-4d59-906c-ffd0e3a596a3.png)

![newplot (1)](https://user-images.githubusercontent.com/94194880/150691576-2c0e3e20-3f11-41fe-863e-4fb607502380.png)

#### File Structure
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

## Dependencies
Machine Learning Libraries: Pandas, Sklearn, Numpy

Natural Language Process Libraries: NLTK

SQLlite Database Libraries: SQLalchemy


