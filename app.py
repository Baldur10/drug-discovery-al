from email import utils
from fileinput import filename
import os
from tkinter.font import names
from turtle import shape
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

import sys
sys.path.insert(0, './scripts')
import nnr_custom

import pickle
import datetime
import json

from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors as rdmd


# Create flask app
flask_app = Flask(__name__,static_folder='./templates/static')
# flask_app = Flask(__name__)

@flask_app.route("/")
def Home():
    if os.path.isdir('data/data_uploads/') == False:
            os.mkdir('data/data_uploads/')

    if os.path.isdir('data/data_output/') == False:
            os.mkdir('data/data_output/')

    return render_template("index.html")

        

@flask_app.route("/result", methods=['POST'])
def result():
    if request.method == 'POST':

        file = request.files['smilesFile']
        params_list = request.form.to_dict()
        filename = secure_filename(file.filename)
        file.save(os.path.join('data/data_uploads/',filename))

        file_content = pd.read_csv(os.path.join('data/data_uploads/',filename), names=['smiles'])

        fingerprint_df = []
        for row in file_content.itertuples():
            mol = Chem.MolFromSmiles(row.smiles)
            temp_fingerprint = rdmd.GetMorganFingerprintAsBitVect(mol, radius=2)
            temp_fingerprint = np.array(temp_fingerprint)
            fingerprint_df.append(temp_fingerprint)
        
        if (params_list['model-select']=='GPR'):
            model_variable = "Matern()"
        elif (params_list['model-select']=='RFR'):
            model_variable = "RandomForestRegressor()"
        elif (params_list['model-select']=='NNR'):
            model_variable = "BasicNNR()"

        model_path = "models/{}_morgan/{}_{}_{}.pickle".format(params_list['model-select'], model_variable, params_list["accFunction-select"], params_list['assayID'])

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except FileNotFoundError as e:
            return render_template('error.html', error=e)

        predictions = model.predict(np.array(fingerprint_df).astype(np.float32))
        predictions_rounded = []
        results = []
        item_count = 0
        
        for element in predictions:
            potency = round(element[0],2) if params_list['model-select']=='NNR' else round(element,2)
            print('potency is of {} and type {}'.format(potency, type(potency)))

            interim_result = {
                "count" : item_count,
                "mol" : '{}'.format(file_content['smiles'].iloc[item_count]),
                "potency" : round(potency.astype(np.float64),2)
            }
            results.append(interim_result)
            predictions_rounded.append(potency)
            item_count+=1
        
        predictions_rounded = pd.DataFrame(predictions_rounded, columns=['output_expt_pIC50'])

        results_df = pd.concat([file_content,predictions_rounded], axis=1)

        current_time = datetime.datetime.now()
        results_df.to_csv('data/data_output/{}_{}.csv'.format(model_variable,current_time), index=False)
        results_df.to_parquet('data/data_output/{}_{}.parquet'.format(model_variable,current_time), index=False)

        return render_template('predictions.html', results = results)

if __name__ == "__main__":
    flask_app.run(debug=True)