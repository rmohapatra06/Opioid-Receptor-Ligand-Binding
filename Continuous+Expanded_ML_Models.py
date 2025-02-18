

import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
 
from warnings import filterwarnings
import time
import math

import numpy as np
from sklearn import svm, metrics, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import auc, accuracy_score, recall_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor



import matplotlib.pyplot as plt



def seed_everything(seed=22):
    """Set the RNG seed in Python and Numpy"""
    import random
    import os
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def calculate_ro5_properties(smiles):
    """
    Test if input molecule (SMILES) fulfills Lipinski's rule of five.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    pandas.Series
        Molecular weight, number of hydrogen bond acceptors/donor and logP value
        and Lipinski's rule of five compliance for input molecule.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate Ro5-relevant chemical properties
    molecular_weight = Descriptors.ExactMolWt(molecule)
    n_hba = Descriptors.NumHAcceptors(molecule)
    n_hbd = Descriptors.NumHDonors(molecule)
    logp = Descriptors.MolLogP(molecule)
    # Check if Ro5 conditions fulfilled
    conditions = [molecular_weight <= 500, n_hba <= 10, n_hbd <= 5, logp <= 5]
    ro5_fulfilled = sum(conditions) >= 3
    # Return True if no more than one out of four conditions is violated
    return pd.Series(
        [molecular_weight, n_hba, n_hbd, logp, ro5_fulfilled],
        index=["molecular_weight", "n_hba", "n_hbd", "logp", "ro5_fulfilled"],
    )

def convert_ic50_to_pic50(IC50_value):
    pIC50_value = 9 - math.log10(IC50_value)
    return pIC50_value


def smiles_to_fp(smiles, method="maccs", n_bits=2048):
    """
    Encode a molecule from a SMILES string into a fingerprint.

    Parameters
    ----------
    smiles : str
        The SMILES string defining the molecule.

    method : str
        The type of fingerprint to use. Default is MACCS keys.

    n_bits : int
        The length of the fingerprint.

    Returns
    -------
    array
        The fingerprint array.

    """

    # convert smiles to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if method == "morgan2":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(fpg.GetFingerprint(mol))
    if method == "morgan3":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
        return np.array(fpg.GetFingerprint(mol))
    else:
        # NBVAL_CHECK_OUTPUT
        print(f"Warning: Wrong method specified: {method}. Default will be used instead.")
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    
def plot_roc_curves_for_models(models, test_x, test_y, save_png=False):

    """
    Helper function to plot customized roc curve.

    Parameters
    ----------
    models: dict
        Dictionary of pretrained machine learning models.
    test_x: list
        Molecular fingerprints for test set.
    test_y: list
        Associated activity labels for test set.
    save_png: bool
        Save image to disk (default = False)

    Returns
    -------
    fig:
        Figure.
    """

    fig, ax = plt.subplots()

    # Below for loop iterates through your models list
    for model in models:
        # Select the model
        ml_model = model["model"]
        # Prediction probability on test set
        test_prob = ml_model.predict_proba(test_x)[:, 1]
        # Prediction class on test set
        test_pred = ml_model.predict(test_x)
        # Compute False postive rate and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(test_y, test_prob)
        # Calculate Area under the curve to display on the plot
        auc = roc_auc_score(test_y, test_prob)
        # Plot the computed values
        ax.plot(fpr, tpr, label=(f"{model['label']} AUC area = {auc:.2f}"))

    # Custom settings for the plot
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    # Save plot
    return fig 


def model_performance(ml_model, test_x, test_y, verbose=True):
    """
    Helper function to calculate model performance

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    test_x: list
        Molecular fingerprints for test set.
    test_y: list
        Associated activity labels for test set.
    verbose: bool
        Print performance measure (default = True)

    Returns
    -------
    tuple:
        Accuracy, sensitivity, specificity, auc on test set.
    """

    # Prediction probability on test set
    # test_prob = ml_model.predict_proba(test_x)[:, 1]

    # Prediction class on test set
    test_pred = ml_model.predict(test_x)

    # Performance of model on test set
    # accuracy = accuracy_score(test_y, test_pred)
    # sens = recall_score(test_y, test_pred)
    # spec = recall_score(test_y, test_pred, pos_label=0)
    
    r2 = r2_score(test_y, test_pred)
    mae = mean_absolute_error(test_y, test_pred)
    mse = mean_squared_error(test_y, test_pred)
    rmse = np.sqrt(mse)

    if verbose:
        # Print performance results
        # NBVAL_CHECK_OUTPUT        print(f"Accuracy: {accuracy:.2}")
        print(f"r2: {r2:.2f}")
        print(f"mae: {mae:.2f}")
        print(f"rmse {rmse:.2f}")
    return r2, mae, rmse


def model_training_and_validation(ml_model, name, splits, verbose=True):
    """
    Fit a machine learning model on a random train-test split of the data
    and return the performance measures.

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    name: str
        Name of machine learning algorithm: RF, SVM, ANN
    splits: list
        List of desciptor and label data: train_x, test_x, train_y, test_y.
    verbose: bool
        Print performance info (default = True)

    Returns
    -------
    tuple:
        Accuracy, sensitivity, specificity, auc on test set.

    """
    train_x, test_x, train_y, test_y = splits

    # Fit the model
    ml_model.fit(train_x, train_y)

    # Calculate model performance results
    r2, mae, rmse = model_performance(ml_model, test_x, test_y, verbose)

    return  r2, mae, rmse

def crossvalidation(ml_model, df, n_folds=5, verbose=False):
    """
    Machine learning model training and validation in a cross-validation loop.

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    df: pd.DataFrame
        Data set with SMILES and their associated activity labels.
    n_folds: int, optional
        Number of folds for cross-validation.
    verbose: bool, optional
        Performance measures are printed.

    Returns
    -------
    None

    """
    t0 = time.time()
    # Shuffle the indices for the k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Results for each of the cross-validation folds
    # acc_per_fold = []
    # sens_per_fold = []
    # spec_per_fold = []
    # auc_per_fold = []
    r2_per_fold = []
    mae_per_fold = []
    rmse_per_fold = []


    # Loop over the folds
    for train_index, test_index in kf.split(df):
        # clone model -- we want a fresh copy per fold!
        fold_model = clone(ml_model)
        # Training

        # Convert the fingerprint and the label to a list
        train_x = df.iloc[train_index].fp.tolist()
        train_y = df.iloc[train_index].active.tolist()

        # Fit the model
        fold_model.fit(train_x, train_y)

        # Testing

        # Convert the fingerprint and the label to a list
        test_x = df.iloc[test_index].fp.tolist()
        test_y = df.iloc[test_index].active.tolist()

        # Performance for each fold
        r2, mae, rmse = model_performance(fold_model, test_x, test_y, verbose)

        # Save results
        # acc_per_fold.append(accuracy)
        # sens_per_fold.append(sens)
        # spec_per_fold.append(spec)
        # auc_per_fold.append(auc)
        r2_per_fold.append(r2)
        mae_per_fold.append(mae)
        rmse_per_fold.append(rmse)

    # Print statistics of results
    print(
        f"Mean r2: {np.mean(r2_per_fold):.2f} \t"
        f"and std : {np.std(r2_per_fold):.2f} \n"
        f"Mean MAE: {np.mean(mae_per_fold):.2f} \t"
        f"and std : {np.std(mae_per_fold):.2f} \n"
        f"Mean RMSE: {np.mean(rmse_per_fold):.2f} \t"
        f"and std : {np.std(rmse_per_fold):.2f} \n"
        
    )

    return r2_per_fold, mae_per_fold, rmse_per_fold



##### Target identification
targets_api = new_client.target
compounds_api = new_client.molecule
bioactivities_api = new_client.activity


### mu opioid receptor antagonist 
uniprot_id = "P35372"

targets = targets_api.get(target_components__accession=uniprot_id).only(
    "target_chembl_id", "organism", "pref_name", "target_type"
    )


targets = pd.DataFrame.from_records(targets)
target = targets.iloc[0] ### our selected target 
chembl_id = target.target_chembl_id # target id


# bioactivity data filter 
bioactivities = bioactivities_api.filter(
    target_chembl_id=chembl_id, type="IC50", relation="=", assay_type="B"
).only(
    "activity_id",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "molecule_chembl_id",
    "type",
    "standard_units",
    "relation",
    "standard_value",
    "target_chembl_id",
    "target_organism",
)


# get bioassay data for the target based on the filter
bioactivities_df = pd.DataFrame.from_dict(bioactivities)

print(bioactivities_df.head(10))

# filter units 
bioactivities_df["units"].unique()
bioactivities_df.drop(["units", "value"], axis=1, inplace=True)

# set to float 
bioactivities_df = bioactivities_df.astype({"standard_value": "float64"})


### drop missing values 
bioactivities_df.dropna(axis=0, how="any", inplace=True)
print(f"DataFrame shape: {bioactivities_df.shape}")

## keep only one unit of measurements 
bioactivities_df = bioactivities_df[bioactivities_df["standard_units"] == "nM"]


## drop duplicates 
bioactivities_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
print(f"DataFrame shape: {bioactivities_df.shape}")


# reset index and rename columns 
bioactivities_df.reset_index(drop=True, inplace=True)
bioactivities_df.rename(
    columns={"standard_value": "IC50", "standard_units": "units"}, inplace=True
)

bioactivities_df = bioactivities_df[bioactivities_df['IC50'] > 0 ]

# obtain chemical data about active compounds 
compounds_provider = compounds_api.filter(
    molecule_chembl_id__in=list(bioactivities_df["molecule_chembl_id"])
).only("molecule_chembl_id", "molecule_structures")

compounds = list(tqdm(compounds_provider))

# dataset of compounds 
compounds_df = pd.DataFrame.from_records(
    compounds,
)
print(compounds_df.shape)


# clean compound dataframe
compounds_df.dropna(axis=0, how="any", inplace=True)
compounds_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
compounds_df.iloc[0].molecule_structures.keys()


# add canonical smiles 
canonical_smiles = []

for i, compounds in compounds_df.iterrows():
    try:
        canonical_smiles.append(compounds["molecule_structures"]["canonical_smiles"])
    except KeyError:
        canonical_smiles.append(None)

compounds_df["smiles"] = canonical_smiles
compounds_df.drop("molecule_structures", axis=1, inplace=True)

# Sanity check: Remove all molecules without a canonical SMILES string.
compounds_df.dropna(axis=0, how="any", inplace=True)
print(f"DataFrame shape: {compounds_df.shape}")


# merge the two dataframes 
molecules = pd.merge(
    bioactivities_df[["molecule_chembl_id", "IC50", "units"]],
    compounds_df,
    on="molecule_chembl_id",
)
molecules.reset_index(drop=True, inplace=True)


molecules["pIC50"] = molecules.apply(lambda x: convert_ic50_to_pic50(x.IC50), axis=1)


#### FILTER DATAFRAME FOR LIPINSKI RULE OF FIVE 


# check for ro5 properties and merge the results with the original dataframe 
ro5_properties = molecules["smiles"].apply(calculate_ro5_properties)
molecules = pd.concat([molecules, ro5_properties], axis=1)


molecules_ro5_fulfilled = molecules[molecules["ro5_fulfilled"]]
molecules_ro5_violated = molecules[~molecules["ro5_fulfilled"]]

print(f"# compounds in unfiltered data set: {molecules.shape[0]}")
print(f"# compounds in filtered data set: {molecules_ro5_fulfilled.shape[0]}")
print(f"# compounds not compliant with the Ro5: {molecules_ro5_violated.shape[0]}")


lipinski_df = molecules_ro5_fulfilled

### FILTER FOR PAINS 
params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog = FilterCatalog(params)


# search for PAINS
matches = []
clean = []
for index, row in tqdm(lipinski_df.iterrows(), total=lipinski_df.shape[0]):
    molecule = Chem.MolFromSmiles(row.smiles)
    entry = catalog.GetFirstMatch(molecule)  # Get the first matching PAINS
    if entry is not None:
        # store PAINS information
        matches.append(
            {
                "chembl_id": row.molecule_chembl_id,
                "rdkit_molecule": molecule,
                "pains": entry.GetDescription().capitalize(),
            }
        )
    else:
        # collect indices of molecules without PAINS
        clean.append(index)

matches = pd.DataFrame(matches)
opioid_data = lipinski_df.loc[clean]  # keep molecules without PAINS





#### Train ML Model 
filterwarnings("ignore")
# Fix seed for reproducible results
SEED = 22
seed_everything(SEED)



opioid_data["active"] = np.zeros(len(opioid_data))

# Mark every molecule as active with an pIC50 of >= 6.3, 0 otherwise
opioid_data.loc[opioid_data[opioid_data.pIC50 >= 6.3].index, "active"] = 1.0

# NBVAL_CHECK_OUTPUT
print("Number of active compounds:", int(opioid_data.active.sum()))
print("Number of inactive compounds:", len(opioid_data) - int(opioid_data.active.sum()))



compound_df = opioid_data.copy()

compound_df["fp"] = compound_df["smiles"].apply(smiles_to_fp)
compound_df.head(3)

fingerprint_to_model = compound_df.fp.tolist()
label_to_model = compound_df.active.tolist()

# Split data randomly in train and test set
# note that we use test/train_x for the respective fingerprint splits
# and test/train_y for the respective label splits
(
    static_train_x,
    static_test_x,
    static_train_y,
    static_test_y,
) = train_test_split(fingerprint_to_model, label_to_model, test_size=0.2, random_state=SEED)
splits = [static_train_x, static_test_x, static_train_y, static_test_y]
# NBVAL_CHECK_OUTPUT
print("Training data size:", len(static_train_x))
print("Test data size:", len(static_test_x))


# Random Forest Model 
param = {
    "n_estimators": 100,  # number of trees to grows
    "criterion": "entropy",  # cost function to be optimized for a split
}

# model_RF = RandomForestClassifier(**param)
# performance_measures = model_training_and_validation(model_RF, "RF", splits)
# models = [{"label": "Model_RF", "model": model_RF}]


# ### Supporting Vector Model 
# model_SVM = svm.SVC(kernel="rbf", C=1, gamma=0.1, probability=True)
# # Fit model on single split
# performance_measures = model_training_and_validation(model_SVM, "SVM", splits)
# models.append({"label": "Model_SVM", "model": model_SVM})


# # ANN Model 
# model_ANN = MLPClassifier(hidden_layer_sizes=(5, 3), random_state=SEED)
# performance_measures = model_training_and_validation(model_ANN, "ANN", splits)
# models.append({"label": "Model_ANN", "model": model_ANN})

# # Plot roc curve
# plot_roc_curves_for_models(models, static_test_x, static_test_y, True)




linear_reg = LinearRegression()
lin_reg_performance_measures = model_training_and_validation(linear_reg, "LINREG", splits)
models = [{"label": "Model_LinearReg", "model": linear_reg}]


# 2. Decision Tree
tree_reg = DecisionTreeRegressor()
tree_reg_performance_measures = model_training_and_validation(tree_reg, "TreeReg", splits)
models.append({"label": "Model_TreeReg", "model": tree_reg})


# 3. Random Forest
rf_reg = RandomForestRegressor()
rf_reg_performance_measures = model_training_and_validation(rf_reg, "RandomForestReg", splits)
models.append({"label": "Model_RFReg", "model": rf_reg})


# 4. Gradient Boosting
gb_reg = GradientBoostingRegressor()
gb_reg_performance_measures = model_training_and_validation(gb_reg, "GradientBoostReg", splits)
models.append({"label": "Model_GBReg", "model": gb_reg})


# 5. Support Vector Machines (SVM)
svm_reg = SVR()
svm_reg_performance_measures = model_training_and_validation(rf_reg, "SVM_Reg", splits)
models.append({"label": "Model_SVMReg", "model": svm_reg})

# 6. Neural Networks (Deep Learning)
from sklearn.neural_network import MLPRegressor
nn_reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
nn_reg_performance_measures = model_training_and_validation(nn_reg, "NN_Reg", splits)
models.append({"label": "Model_NNReg", "model": nn_reg})

# 7. K-Nearest Neighbors (KNN)
knn_reg = KNeighborsRegressor()
knn_reg_performance_measures = model_training_and_validation(knn_reg, "KNN_Reg", splits)
models.append({"label": "Model_KNNReg", "model": knn_reg})

# 8. Ridge Regression
ridge_reg = Ridge()
ridge_reg_performance_measures = model_training_and_validation(ridge_reg, "Ridge_Reg", splits)
models.append({"label": "Model_RidgeReg", "model": ridge_reg})

# 9. Lasso Regression
lasso_reg = Lasso()
lasso_reg_performance_measures = model_training_and_validation(lasso_reg, "Lasso_Reg", splits)
models.append({"label": "Model_RidgeReg", "model": ridge_reg})

# 10. XGBoost
xgb_reg = XGBRegressor()
xgb_reg_performance_measures = model_training_and_validation(xgb_reg, "XGB_Reg", splits)
models.append({"label": "Model_XGBReg", "model": xgb_reg})



###CROSS VALIDATION of the model 
N_FOLDS = 3

for model in models:
    print("\n======= ")
    print(f"{model['label']}")
    crossvalidation(model["model"], compound_df, n_folds=N_FOLDS)



# unseen_compounds = []

# with gzip.open('ranked_list_top10K_commercial_cmps.dat.gz', 'rt') as file:
#     for line in file:
#         if line[0] == "#":
#             continue
#         line = line.rstrip().split()
#         # contains: [identifier, SMILES, max rank, max proba, similarity]
        
#         unseen_compounds.append(line[1])
        



# commercial_compounds = pd.DataFrame(unseen_compounds, columns=['SMILES'])
# commercial_compounds['fingerprint'] = commercial_compounds['SMILES'].apply(smiles_to_fp)





# Generate predictions for the new data
# RF_model_predictions = model_RF.predict(commercial_compounds['fingerprint'].to_list())
# SVM_model_predictions = model_SVM.predict(commercial_compounds['fingerprint'].to_list())
# ANN_model_predictions = model_ANN.predict(commercial_compounds['fingerprint'].to_list())



# commercial_compounds['RF_Pred'] = RF_model_predictions
# commercial_compounds['SVM_Pred'] = SVM_model_predictions
# commercial_compounds['ANN_Pred'] = ANN_model_predictions


# commercial_compounds.to_csv('labeled.csv')


# active3 = commercial_compounds[(commercial_compounds['RF_Pred'] == 1) 
#                                & (commercial_compounds['SVM_Pred'] == 1) 
#                                & (commercial_compounds['ANN_Pred'] == 1)
# ]

# active3.to_csv('active3.csv')


# docked_results = pd.read_csv("hit_expansion/similarity/expansion_SIM_results.csv")
# docked_results['fingerprint'] = docked_results['smiles'].apply(smiles_to_fp)

# RF_dr_pred = model_RF.predict(docked_results['fingerprint'].to_list())
# SVM_dr_pred = model_SVM.predict(docked_results['fingerprint'].to_list())
# ANN_dr_pred = model_ANN.predict(docked_results['fingerprint'].to_list())



# docked_results['RF_Pred'] = RF_dr_pred
# docked_results['SVM_Pred'] = SVM_dr_pred
# docked_results['ANN_Pred'] = ANN_dr_pred


# docked_results.to_csv('first_query_predicted.csv')

