from utils.calc.mof_collection import MofCollection
import os, shutil, stat

def remove_dir_with_permissions(dir_path):
    def handle_permission_error(func, path, exc_info):
        os.chmod(path, stat.S_IWUSR)
        func(path)

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, onerror=handle_permission_error)

def get_from_file(struc):
    a_mof_collection = MofCollection(path_list = [struc], 
                                 analysis_folder="tmp_oms")
    a_mof_collection.analyse_mofs(num_batches=1,overwrite=False)
    oms_result = {
                    "Metal Types": a_mof_collection.mof_oms_df["Metal Types"][struc.replace(".cif","").split("/")[-1]],
                    "Has OMS": a_mof_collection.mof_oms_df["Has OMS"][struc.replace(".cif","").split("/")[-1]],
                    "OMS Types": a_mof_collection.mof_oms_df["OMS Types"][struc.replace(".cif","").split("/")[-1]],
                }

    remove_dir_with_permissions("tmp_oms")

    return oms_result
