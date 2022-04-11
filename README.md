# Active Learning Systems for Screening Drugs
***
## _City University of Hong Kong_
### _Final Year Project 2021-22_
### _Supervisor: Dr.WEI, Ying_
***
# Description
This project is an Active Learning Framework for Drug Discovery which has been designed to be extendable and permits the integration of different Regression models for determing the potency of chemcial compounds.

# Prerequisites
1. Python Environment Manager such as conda or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Code Editor - [Visual Studio Code](https://code.visualstudio.com/) or [Jupyter-Lab](https://jupyter.org/install) preferred

# Usage
1. Clone repo into the selected folder via 
    ```sh 
    git clone https://gitlab.com/Baldur10/drug-discovery-al
    ```
2. Enter the root folder of the repo via 
    ```sh 
    cd \drug-discovery-al\
    ```
3. Set up the conda environment by 
    ```sh
    conda create --name dd-al --file=environ_al.yml
    ```
4. Activate the conda environment via
    ```sh
    conda activate dd-al
    ```
# ML Models Available
1. Gaussian Processes Regressor ([Scikit-Learn GPR](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html))
2. Random Forest Regressor ([Intel(R) Extension for Scikit-Learn ](https://intel.github.io/scikit-learn-intelex/algorithms.html) and [Scikit-Learn RFR](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html))
3. Neural Network Regressor ([SKORCH](https://github.com/skorch-dev/skorch))

Pretrained Models for the default assays are available at:

| Storage | Link |
|----|----|
|Onedrive|[FYP Models](https://portland-my.sharepoint.com/:f:/g/personal/rmohan2-c_my_cityu_edu_hk/Eui7FaFOAStKmofPVNrHDzQB9om1OmAXD2aK_RpYDRHJUg?e=wB3ZC5)|

# ML Loops
## Training Loop
1. Open the requisite model training scripts inside `/scripts`
2. Taking the example of the Gaussian Processes Regressor Model, the approriater file is `/scripts/test_gpr.ipynb`
3. Open the file in the code editor and run all cells. If given the option, select `dd-al` as the python interpretator
4. The variable `assay_limit` can be changed to any integer 'n' to set the first 'n' number of assays for which models have to be trained.
5. After the training loop is completed, the models can be found under `/models` and the data is present under `/data/data_results`

## Testing Loop
1. Before running Flask, set the environment variable using 
    ```sh
    set FLASK_APP=app.py
    ```
2. Run the Flask app via 
    ```sh 
    flask run
    ```
# Support
Contact me at rmohan2-c@my.cityu.edu.hk




