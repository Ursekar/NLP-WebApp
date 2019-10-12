# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python train_classifier.py DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Additive information:
1.) disaster_categories.csv and disaster_messages.csv (dataset).

2.) DisasterResponse.db: created database from transformed and cleaned data.

3.) train_classifier.py: includes the code necessary to load data, transform it using natural language processing, run a machine learning model using GridSearchCV and train it

4.) run.py: Flask app and the user interface used to predict results and display them.

### Web App:
1.) Just type in your message in textbox provided.

2.) Click on "Classify Message" button or hit Enter.

3.) The green highleted categories appearing beneath are your class(es) of typed in message.
