import os
import zipfile
import pandas as pd

def make_dir_if_not_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def unzipfiles(list_inputs, directory, remove = True):
    local_zips = [ os.path.join(directory, x) for x in list_inputs]

    for local_zip in local_zips:
        if not os.path.exists(local_zip):
            print(f'file {local_zip} doesn''t exists!')
        else :
            zip_ref = zipfile.ZipFile(local_zip, 'r')
            zip_ref.extractall(directory)
            zip_ref.close()
        if remove == True:
            os.remove(local_zip)


def export_dataframe_csv(df, path):
    make_dir_if_not_exists(path)
    df.to_csv(path, index=False)


def csv_to_df(input_file, sep=',', names=None, header=None):
    return pd.read_table(input_file, sep=sep, header=header, names=names)

