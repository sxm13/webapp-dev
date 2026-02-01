import os, warnings
import pandas as pd
from utils.pred.cp_app.descriptors import cv_features
from utils.pred.cp_app.featurizer import featurize_structure
from utils.pred.cp_app.predictions import predict_Cv_ensemble_structure_multitemperatures
warnings.filterwarnings('ignore')

package_directory = os.path.abspath(__file__).replace("cp.py","")

def run(struc, name, T=[300, 350, 400]):

    featurize_structure(struc, verbos=False, saveto="./data/tmp_files/"+name+"_features.csv")
    
    predict_Cv_ensemble_structure_multitemperatures(
                                                        path_to_models=package_directory+"/cp_app/ensemble_models_smallML_120_100",
                                                        structure_name= name+".cif",
                                                        features_file="./data/tmp_files/"+name+"_features.csv", 
                                                        FEATURES=cv_features,
                                                        temperatures=T,
                                                        save_to="./data/tmp_files/"+name+"_cp.csv"
                                                    )
    result_ = pd.read_csv("./data/tmp_files/"+name+"_cp.csv")
    result_cp = {}
    result_cp["unit"] = "J/g/K", "J/mol/K"

    for t in T:
        result_cp[str(t)+"_mean"] = [result_["Cv_gravimetric_"+str(t)+"_mean"].iloc[0],
                                        result_["Cv_molar_"+str(t)+"_mean"].iloc[0]]
        result_cp[str(t)+"_std"] = [result_["Cv_gravimetric_"+str(t)+"_std"].iloc[0],
                                        result_["Cv_molar_"+str(t)+"_std"].iloc[0]]

    return result_cp