import pandas as pd
import numpy as np
import sqlite3
import sys
import sqlalchemy as sqlac

def load_data(messages_filepath, categories_filepath):
    
    # Load Messages Dataset
    messages = pd.read_csv(str(messages_filepath)) #'messages.csv'
    print('The shape of your messages data is: ', messages.shape)
    print(messages.head())   
    
    # Load Categories Dataset
    categories = pd.read_csv(str(categories_filepath)) #'categories.csv'
    print('\n The shape of your messages data is: ',categories.shape)
    print(categories.head())
    
    # Merge two dataframes
    df = pd.merge(messages,categories,on = 'id')
    print('\n Have a look at your complete dataframe')
    print(df.head())
    
    return df

def clean_data(df):
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    pd.set_option('display.max_colwidth', -1) # For displaying entire columns of df without truncating
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x.split('-')[0] for x in row]
    print('\n These are the category names which will be column names', category_colnames)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()
    
    # Convert each category to numeric with values 1 or 0
    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = [row_val.split('-')[1] for row_val in categories[str(column)]]
    
        # convert column from string to numeric
        categories[column] = [int(row_val) for row_val in categories[str(column)]]

    print('\n View of numeric conversion of categories: ', categories.head())
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    print('\n dropping categories column', df.head())
          
    # concatenate the original dataframe with the new `categories` dataframe
    df.reset_index(drop = True, inplace= True)              
    categories.reset_index(drop = True, inplace= True)
    df = pd.concat([df,categories], axis = 1)
    print('\n Concatenated df: ', df.head())
    
    # check number of duplicates
    print('\ df before dropping duplicates: ', df.shape)
    df.drop_duplicates(inplace = True)
    print('\n df after dropping duplicates: ', df.shape)
    
    return df

def save_data(df, database_filename):
    
    engine = sqlac.create_engine('sqlite:///'+str(database_filename)) #'sqlite:///InsertDatabaseName.db'
    df.to_sql('disaster_table', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()