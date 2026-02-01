import streamlit as st
from streamlit_extras.badges import badge


st.set_page_config(
                    page_title="CoRE MOF API",
                    page_icon="⚙",
                    layout="centered"
                    )


st.title("API")

st.divider()

logo_url = "./figures/logo.png"
st.sidebar.image(logo_url)
st.logo(logo_url, link="https://zenodo.org/communities/core-mofs/records?q=&l=list&p=1&s=10&sort=newest")

with st.sidebar:
    st.markdown("© 2024 CoRE MOF Project. CC-BY-4.0 License.")

badge(type="pypi", name="CoREMOF-tools")

st.markdown("## Installation")
st.markdown("Install by `pip`")
st.code("pip install CoREMOF-tools")
st.code("conda install conda-forge::zeopp-lsmo")

st.markdown("[Document for CoREMOF_tools](https://coremof-tools.readthedocs.io/en/latest/genindex.html)")

st.divider()

st.markdown("## Use Examples")

st.markdown("### Download CoRE MOF Database")
st.markdown("- Download CIFs from SI (include CR and NCR), do not need to process CIFs due to open source")
st.markdown("""
            ```python
            from CoREMOF.structure import download_from_SI
            download_from_SI(output_folder='./CoREMOF2024DB')
            """)

st.markdown("- For CIFs from CSD, you need to install CSD software with license first, there are two list of CR and NCR dataset")
st.markdown("""
            ```python
            from CoREMOF.structure import get_list_CSD, download_from_CSD
            """)

st.markdown("CR contain 'ASR', 'FSR', 'Ion' set, and there is CoRE ID and CSD REFCODE information.")
st.markdown("""
            ```python
            CR = get_list_CSD()[0]
            print(CR.keys())
            for file in CR["ASR"]:
                print(file)
            """)

st.markdown("NCR contain 'both', 'Chen_Manz', 'mofchecker', 'MOSAEC' set, and there is CSD REFCODE information.")
st.markdown("""
            ```python
            NCR = get_list_CSD()[1]
            print(NCR.keys())
            for file in NCR["both"][:10]:
                print(file)
            """)

st.markdown("- Download structure from CSD (original)")
st.markdown("""
            ```python
            download_from_CSD(refcode=CR["ASR"][0][1].split("_")[0], output_folder="./CoREMOF2024DB/CR")
            """)

st.markdown("- process original CIF to CoRE MOF")
st.markdown("""
            ```python
            from CoREMOF.structure import make_primitive_p1
            structure_pri = make_primitive_p1(filename="./CoREMOF2024DB/CR/ABAVIJ.cif") # make primitive cell and P1

            from CoREMOF.curate import clean
            clean(structure="./CoREMOF2024DB/CR/ABAVIJ.cif", output_folder="./CoREMOF2024DB/CR/", saveto=None) # remove free and coordinated solvent

            from CoREMOF.prediction import pacman
            results_eb = pacman(structure="./CoREMOF2024DB/CR/ABAVIJ_ASR.cif", output_folder="./CoREMOF2024DB/CR/") # assign partial charge
            
            import os
            os.rename("./CoREMOF2024DB/CR/" + CR["ASR"][0][1]+".cif", "./CoREMOF2024DB/CR/" + CR["ASR"][0][0]+".cif")  # change REFCODE to CoRE ID: ABAVIJ_ASR_pacman - > 2004[Co][rtl]3[ASR]1
            """)


st.markdown("### Query information for CoRE MOF DB")
st.markdown("""preprocess CIF (check multi structure, metal, carbon. make primitive cell. make P1)""")
st.markdown("""entry:CR: CoRE ID; NCR: REFCODE""")
                
st.markdown("""
            ```python
            from CoREMOF.structure import information
            data = information("CR-ASR", "2020[Cu][sql]2[ASR]1")') # CR
            data = information("NCR", "10853_2020_5211_MOESM2_ESM_ASR_pacman") # NCR
            """)

st.markdown("### Curate and check structure")

st.markdown("""
            ```python
            from CoREMOF.curate import preprocess, clean, mof_check
            """)

st.markdown("""- preprocess CIF""")

                
st.markdown("""
            ```python
            preprocess(structure="./CoREMOF2024DB/example/c7dt03758a2.cif", output_folder="CoREMOF2024DB/result_curation/")
            """)

st.markdown("""- solvent removal""")
st.markdown("""
            ```python
            clean(structure="CoREMOF2024DB/result_curation/c7dt03758a2_1.cif", initial_skin = 0.25, output_folder="CoREMOF2024DB/result_curation", saveto="clean_result.csv")
            """)

st.markdown("""- NCR classification""")
st.markdown("""
            ```python
            mof_check(structure="CoREMOF2024DB/result_curation/c7dt03758a2_1.cif", output_folder="CoREMOF2024DB/result_curation")
            """)


st.markdown("### Zeopp (Zeo++) calculation")

st.markdown("""
            ```python
            from CoREMOF.calculation import Zeopp
            """)

st.markdown("""- Pore diameter""")
st.markdown("""
            ```python
            results_pd = Zeopp.PoreDiameter(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif", high_accuracy = True)
            print(results_pd)
            """)

st.markdown("""- Surface area""")
st.markdown("""
            ```python
            results_sa = Zeopp.SurfaceArea(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif",
                               chan_radius = 1.655,
                               probe_radius = 1.655,
                               num_samples = 5000,
                               high_accuracy = True)
            print(results_sa)
            """)

st.markdown("""- Pore volume""")
st.markdown("""
            ```python
            results_pv = Zeopp.PoreVolume(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif",
                              chan_radius = 0,
                              probe_radius = 0,
                              num_samples = 5000,
                              high_accuracy = True)
            print(results_pv)
            """)

st.markdown("""- Dimension for channel""")
st.markdown("""
            ```python
            results_chan = Zeopp.ChanDim(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif",
                             probe_radius = 0,
                             high_accuracy = True)
            print(results_chan)
            """)

st.markdown("""- Dimension for framework""")
st.markdown("""
            ```python
            results_strinfo = Zeopp.FrameworkDim(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif", high_accuracy = True)
            print(results_strinfo)
            """)


st.markdown("### Machine learning features prediction")

st.markdown("""
            ```python
            from CoREMOF.prediction import cp, stability
            """)

st.markdown("""- Heat capacity""")
st.markdown("""
            ```python
            result_cp = cp(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif", T=[300, 350, 400])
            print(result_cp)
            """)

st.markdown("""- Stabilities""")
st.markdown("""
            ```python
            result_s = stability(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif")
            print(result_s)
            """)


st.markdown("### Other features")

st.markdown("""
            ```python
            from CoREMOF.calculation import mof_features
            """)

st.markdown("""- Space group""")
st.markdown("""
            ```python
            result_sg=mof_features.SpaceGroup(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif")
            print(result_sg))
            """)

st.markdown("""- Mass""")
st.markdown("""
            ```python
            result_m=mof_features.Mass(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif")
            print(result_m)
            """)

st.markdown("""- Volume""")
st.markdown("""
            ```python
            result_v=mof_features.Volume(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif")
            print(result_v)
            """)

st.markdown("""- Number of atoms""")
st.markdown("""
            ```python
            result_na=mof_features.n_atom(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif")
            print(result_na)
            """)

st.markdown("""- Topology""")
st.markdown("""
            ```python
            result_topo = mof_features.topology(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif", node_type="single")
            print(result_topo)
            """)

st.markdown("""- Revised autocorrelation""")
st.markdown("""
            ```python
            result_rac = mof_features.RACs(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif")
            print(result_rac)
            """)

st.markdown("""- Open metal site""")
st.markdown("""
            ```python
            result_oms = mof_features.get_oms_file(structure="CoREMOF2024DB/CR/2004[Co][rtl]3[ASR]1.cif")
            print(result_oms)
            """)

st.markdown("""### Dependencies""")
st.markdown("""
| Package                      | Version    |
|------------------------------|------------|
| ase                          | 3.24.0     |
| joblib                       | 1.4.2      |
| keras                        | 3.8.0      |
| markdownlit                  | 0.0.7      |
| matminer                     | 0.9.3      |
| matplotlib                   | 3.9.4      |
| mofchecker                   | 0.9.6      |
| molSimplify                  | 1.7.6      |
| networkx                     | 3.2.1      |
| numpy                        | 2.0.2      |
| openbabel-wheel              | 3.1.1.21   |
| PACMAN-charge                | 1.3.9      |
| plotly                       | 6.0.0      |
| pymatgen                     | 2024.8.9   |
| scikit-learn                 | 1.6.1      |
| scipy                        | 1.13.1     |
| stmol                        | 0.0.9      |
| streamlit                    | 1.2.3      |
| streamlit-extras             | 0.5.0      |
| tensorflow                   | 2.18.0     |
| torch                        | 2.6.0      |
| xgboost                      | 2.1.4      |
""")