import pandas as pd
from datetime import datetime
import os

current_dir = os.getcwd()
current_datetime = datetime.now()
time_now = current_datetime.strftime('%d%m%Y%H%M%S')


def filename_formatter(filename, _order):    
    _filename=os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    order = True
    if _order == 'A':
        order = True
        _filename = _filename + '_ASC_'
    elif _order == 'D':
        order = False
        _filename = _filename + '_DSC_'
    filename = _filename + time_now + ext
    return filename, order


def convert_order(_order):
    order = True
    if _order == 'A':
        order = True
    elif _order == 'D':
        order = False
    return order


# Copy downloaded data in the csv format to working directory
def save_df_to_csv(_df, filename, _order,target_dir):
    #file name with extension .csv
    #order 'A' for Ascending 'D' for descending
    filename, order = filename_formatter(filename, _order)

    _df = _df.sort_values(by=['Stock', 'Date'], ascending=[True, order])

    try:
        cd = os.getcwd()
        sd = os.path.join(cd, target_dir)
        os.chdir(sd)
        _df.to_csv(filename, float_format='%.5f')
        print(f"File saved successfully as {filename}")
        os.chdir(cd)
        
    except Exception as e:
        print(f"An error occurred: {e}")


# Copy downloaded data in the .pkl format to working directory
def save_df_to_pkl(_df, filename, _order):
    #file name with extension .pkl
    #order 'A' for Ascending 'D' for descending
    filename, order = filename_formatter(filename, _order)
    _df = _df.sort_values(by=['Stock', 'Date'], ascending=[True, order])

    try:
        cd = os.getcwd()
        sd = os.path.join(cd, 'sub_dir')
        os.chdir(sd)
        _df.to_pickle(filename)
        print(f"File saved successfully as {filename}")
        os.chdir(cd)
    except Exception as e:
        print(f"An error occurred: {e}")


def read_csv_to_df(filename, _order,source_dir):
    # make 'files' as the current working directory
    # read  *.csv/sub_dir
    cd = os.getcwd()
    sd = os.path.join(cd, source_dir)
    os.chdir(sd)

    order = convert_order(_order)
    _df = pd.read_csv(filename)
    _df['Date'] = pd.to_datetime(_df['Date'])
    _df.set_index('Date', inplace=True)  # set Date column as index
    _df.sort_index(axis=0, inplace=True)  # index based sorting compulsory for slicing
    _df = _df.sort_values(by=['Stock', 'Date'], ascending=[True, order])

    os.chdir(cd)

    return _df


def read_pkl_to_df(filename, _order):
    order = convert_order(_order)
    _df = pd.read_pickle(filename)
    _df['Date'] = pd.to_datetime(_df['Date'])
    _df.set_index('Date', inplace=True)  # set Date column as index
    _df.sort_index(axis=0, inplace=True)  # index based sorting compulsory for slicing
    _df = _df.sort_values(by=['Stock', 'Date'], ascending=[True, order])
    return _df


def convert_csv_to_pkl(filename_csv, filename_pkl, _order):
    df = read_csv_to_df(filename_csv, _order)
    save_df_to_pkl(df, filename_pkl, _order)


def convert_pkl_to_csv(filename_csv, filename_pkl, _order):
    df = read_pkl_to_df(filename_pkl, _order)
    save_df_to_csv(df, filename_csv, _order)


def change_cwd(sd):
    # print("Current Directory:", current_dir)
    sub_dir = os.path.join(current_dir, sd)
    os.chdir(sub_dir)
    # print("Changed to Subdirectory:", os.getcwd())


def get_cwd():
    os.chdir(current_dir)
    # print("Changed to Current directory:", os.getcwd())


# This code is exclusively made for reading from .csv file and format ,filter,sort etc
def df_slicer(_df, _order,start_date,end_date):
    order = convert_order(_order)    
    _df = _df.loc[:,:]
    # index based sorting compulsory for slicing
    _df.sort_index(axis=0, inplace=True)
    _df = _df.loc[start_date:end_date]
    _df = _df.sort_values(by=['Stock', 'Date'], ascending=[True, order])
    return _df
