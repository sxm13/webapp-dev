import pickle as pkl
import cloudpickle
from utils.calc.pore import SurfaceArea
from utils.calc.RAC import get

def run(structure):
        
    water_feature_names = ['mc-Z-3-all', 'D_mc-Z-3-all', 'D_mc-Z-2-all',
                            'D_mc-Z-1-all', 'mc-chi-3-all', 'mc-Z-1-all',
                            'mc-Z-0-all', 'D_mc-chi-2-all', 'f-lig-Z-2',
                            "ASA",
                            'f-lig-I-0', 'func-S-1-all']

    result_stability = {}
    result_stability["unit"] = "nan"
    
    with open('utils/pred/models/water_model.pkl', 'rb') as f:
        water_model = cloudpickle.load(f)
    with open('utils/pred/models/water_scaler.pkl', 'rb') as f:
        water_scaler = pkl.load(f)

    results_sa_1_4 = SurfaceArea(structure, 1.4, 1.4, 10000, True)
    result_RACs = get(structure)
    
    X_water = []
    for fn_water in water_feature_names:
        try:
            X_water.append(result_RACs["Metal"][fn_water])
        except:
            try:
                X_water.append(result_RACs["Linker"][fn_water])
            except:
                try:
                    X_water.append(result_RACs["Function-group"][fn_water])
                except:
                    X_water.append(results_sa_1_4[fn_water][2])

    X_water = water_scaler.transform([X_water])

    water_model_prob = water_model.predict_proba(X_water)[:,1]
    result_stability["water probability"] = float(water_model_prob[0])

    return result_stability