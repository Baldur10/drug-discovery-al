from email import utils
from fileinput import filename
import os
from tkinter.font import names
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

import pickle
import time

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
        # try:
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
        model_path = "models/{}_morgan/{}_{}_{}.pickle".format(params_list['model-select'], model_variable, params_list["accFunction-select"], params_list['assayID'])

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except FileNotFoundError as e:
            return render_template('error.html', error=e)

        predictions = model.predict(np.array(fingerprint_df))
        results = []
        item_count = 0

        for element in predictions:
            interim_result = {
                "mol" : '{}'.format(file_content['smiles'].iloc[item_count]),
                "potency" : round(element,2)
            }
            results.append(interim_result)
            item_count+=1
        
        results_df = pd.concat(file_content,pd.DataFrame(round(predictions,2), columns=['output_expt_pIC50']), axis=0)
        print('\n',results_df,'\n')
        
        results_df.to_csv('data/data_output/{}.csv'.format(filename), index=False)
        results_df.to_parquet('data/data_output/{}.parquet'.format(filename), index=False)
        
        return render_template('predictions.html', results = results)

if __name__ == "__main__":
    flask_app.run(debug=True)