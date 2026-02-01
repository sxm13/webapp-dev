import os
import subprocess

def ChanDim(struc, probe_radius = 0, high_accuracy = True):
    results_chan = {}
    results_chan["unit"]="nan"
    
    if high_accuracy:
        cmd = f'network -ha -chan {probe_radius} ./data/tmp_files/tmp_chan.txt {struc}'
    else:
        cmd = f'network -chan {probe_radius} ./data/tmp_files/tmp_chan.txt {struc}'
    _ = subprocess.run(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                    )

    with open('./data/tmp_files/tmp_chan.txt') as f:
        for i, row in enumerate(f):
            if i == 0:
                try:
                    dim = int(row.split('dimensionality')[1].split()[0])
                except:
                    dim = 0

    results_chan["Dimention"] = dim

    os.remove("./data/tmp_files/tmp_chan.txt")

    return results_chan

def FrameworkDim(struc, high_accuracy = True):
    results_strinfo = {}
    results_strinfo["unit"]="nan"
    
    if high_accuracy:
        cmd = f'network -ha -strinfo ./data/tmp_files/tmp_strinfo.txt {struc}'
    else:
        cmd = f'network -strinfo ./data/tmp_files/tmp_strinfo.txt {struc}'
    _ = subprocess.run(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                    )

    with open('./data/tmp_files/tmp_strinfo.txt') as f:
        line = f.readline().split()
        try:
            dim = int(line[-1])
            one_dim = int(line[7])
            two_dim = int(line[8])
            three_dim = int(line[9])
        except:
            one_dim = 0
            two_dim = 0
            three_dim = 0
            dim = 0
    results_strinfo["Dimention"] = dim
    results_strinfo["N_1D"] = one_dim
    results_strinfo["N_2D"] = two_dim
    results_strinfo["N_3D"] = three_dim

    os.remove("./data/tmp_files/tmp_strinfo.txt")

    return results_strinfo