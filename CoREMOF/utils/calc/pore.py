import os
import subprocess

def PoreDiameter(struc, high_accuracy = True):

    results_pd = {}
    results_pd["unit"]="angstrom, Å"
    
    if high_accuracy:
        cmd = f'network -ha -res ./data/tmp_files/tmp_pd.txt {struc}'
    else:
        cmd = f'network -res ./data/tmp_files/tmp_pd.txt {struc}'
    _ = subprocess.run(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                    )
    with open('./data/tmp_files/tmp_pd.txt') as f:
        line = f.readline().split()
        results_pd["LCD"], results_pd["PLD"], results_pd["LFPD"] = map(float, line[1:4])
    os.remove('./data/tmp_files/tmp_pd.txt')

    return results_pd

def SurfaceArea(struc, chan_radius = 1.655, probe_radius = 1.655, num_samples = 5000, high_accuracy = True):
    results_sa = {}
    results_sa["unit"]="Å^2, m^2/cm^3, m^2/g"
    
    if high_accuracy:
        cmd = f'network -ha -sa {chan_radius} {probe_radius} {num_samples} ./data/tmp_files/tmp_sa.txt {struc}'
    else:
        cmd = f'network -sa {chan_radius} {probe_radius} {num_samples} ./data/tmp_files/tmp_sa.txt {struc}'
    _ = subprocess.run(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                    )
    with open('./data/tmp_files/tmp_sa.txt') as f:
        for i, row in enumerate(f):
            if i == 0:
                ASA = float(row.split('ASA_A^2:')[1].split()[0])
                VSA = float(row.split('ASA_m^2/cm^3:')[1].split()[0])
                GSA = float(row.split('ASA_m^2/g:')[1].split()[0])
                NASA = float(row.split('NASA_A^2:')[1].split()[0])
                NVSA = float(row.split('NASA_m^2/cm^3:')[1].split()[0])
                NGSA = float(row.split('NASA_m^2/g:')[1].split()[0])

    results_sa["ASA"] = [ASA, VSA, GSA]
    results_sa["NASA"] = [NASA, NVSA, NGSA]

    os.remove("./data/tmp_files/tmp_sa.txt")

    return results_sa

def PoreVolume(struc, chan_radius = 0, probe_radius = 0, num_samples = 5000, high_accuracy = True):
    results_pv = {}
    results_pv["unit"]="PV: Å^3, cm^3/g; VF: nan"
    
    if high_accuracy:
        cmd = f'network -ha -volpo {chan_radius} {probe_radius} {num_samples} ./data/tmp_files/tmp_pv.txt {struc}'
    else:
        cmd = f'network -volpo {chan_radius} {probe_radius} {num_samples} ./data/tmp_files/tmp_pv.txt {struc}'
    _ = subprocess.run(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                    )
    with open('./data/tmp_files/tmp_pv.txt') as f:
        for i, row in enumerate(f):
            if i == 0:
                POAV = float(row.split('POAV_A^3:')[1].split()[0])
                PONAV = float(row.split('PONAV_A^3:')[1].split()[0])
                GPOAV = float(row.split('POAV_cm^3/g:')[1].split()[0])
                GPONAV = float(row.split('PONAV_cm^3/g:')[1].split()[0])
                POAV_volume_fraction = float(row.split('POAV_Volume_fraction:')[1].split()[0])
                PONAV_volume_fraction = float(row.split('PONAV_Volume_fraction:')[1].split()[0])
    results_pv["PV"] = [POAV, GPOAV]
    results_pv["NPV"] = [PONAV, GPONAV]
    results_pv["VF"] = POAV_volume_fraction
    results_pv["NVF"] = PONAV_volume_fraction

    os.remove("./data/tmp_files/tmp_pv.txt")

    return results_pv
