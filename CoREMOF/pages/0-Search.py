import pandas as pd
import streamlit as st
from stmol import *
from streamlit_extras.capture import stdout
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stoggle import stoggle
from streamlit_extras.markdownlit import mdlit

import warnings
from io import BytesIO, StringIO

from streamlit_extras.tags import tagger_component
from streamlit_extras.toggle_switch import st_toggle_switch
from st_aggrid import AgGrid, GridOptionsBuilder

from ase.io import read, write



warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Search", page_icon="🌟",layout="centered")

st.title("Search")

st.divider()

logo_url = "./figures/logo.png"
st.sidebar.image(logo_url)
st.logo(logo_url, link="https://zenodo.org/communities/core-mofs/records?q=&l=list&p=1&s=10&sort=newest")

with st.sidebar:
    st.markdown("© 2025 CoRE MOF Project. CC-BY-4.0 License.")

mdlit(
        """KEY: You can enter `key` to find structure and information, such as core-id, MOF name, DOI, metal cluster and so on.
        """
    )

# mdlit(
#         """INFO: You can filter by cutoff or specific criteria.
#         """
#     )
st.text("")
st.text("")
st.text("")
st.text("")


def vis(_cif):
    viewer = py3Dmol.view(width=800, height=500)
    atoms = read(StringIO(_cif), format="cif")
    cif_io = BytesIO()
    write(cif_io, atoms, format="cif")
    cif_content = cif_io.getvalue().decode('utf-8')
    viewer.addModel(cif_content, "cif")
    viewer.setStyle({'stick': {}})
    viewer.addUnitCell()
    viewer.zoomTo()
    showmol(viewer, width=800, height=500)


colored_header(
                label="Search from CR Dataset",
                description="by KEY or INFO",
                color_name="blue-70",
                )

st.markdown('''#### :rainbow[by KEY]''')

CR_data = pd.read_csv("./data/internal/All_data_20241205.csv", low_memory=False)
# df = pd.read_csv("./data/internal/All_data_20241205_search_SI.csv", low_memory=False)
CR_search = pd.read_csv("./data/internal/All_data_20241205_search.csv", low_memory=False)


def search_info(searchterm: str) -> list:
    if not searchterm:
        return []
    CR_key = CR_search[CR_search.apply(lambda row: searchterm.lower() in row.astype(str).str.lower().to_string(), axis=1)]
    return CR_key.to_dict(orient="records")
searchterm = st.text_input("Search in CoRE MOF CR database...", key="search_key_cr")
if searchterm:
    search_key_results = search_info(searchterm)
    n_results_cr = len(search_key_results)
    
    st.markdown(f"There are :red[{n_results_cr}] MOFs found!")
    if n_results_cr > 0:
        coreid_ = st.selectbox(
            "Select a MOF by CoRE ID:",
            [row["coreid"] for row in search_key_results]
        )
        coreid_info = next((row for row in search_key_results if row["coreid"] == coreid_), None)
        if coreid_info:
            try:
                type_ = coreid_info["Extension"]
                name = coreid_info["coreid"]
                with open(f"./data/internal/CoREMOF2024DB/CR/{type_}/{name}.cif", "r") as f:
                    cif_content = f.read()
                vis(cif_content)
                data0_ = CR_data[CR_data["coreid"] == coreid_info["coreid"]]
                st.json(data0_.to_dict(orient="records"))
                # st.dataframe(data0_)
                if coreid_info["DOI"] != "unknown":
                    st.markdown(f"[Read Original Paper](https://doi.org/{coreid_info['DOI']})", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error visualizing MOF: {e}")
else:
    pass



st.markdown('''#### :rainbow[by INFO]''')

if st_toggle_switch:
    # col1, col2 = st.columns(2)
    st.markdown("**Supported Filters**")
    # with col1:
    tagger_component("Type:", ["Extension for CR"])
    tagger_component("Source:", ["Journal", "Year"], # "CSD or SI", 
                    color_name=["lightblue", "lightblue"]) # "lightblue", 
    # with col2:
    tagger_component("Geometry:", ["PLD", "LCD", "LFPD", "ASA", "VF", "PV", "Density", "Dimension", "Topology", "N_atoms", "Has OMS", "Metal"],
                    color_name=["blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "orange", "orange"])

    tagger_component("Stability:", ["Thermal Stability", "Solvent Stability", "Water Stability", "Hydrophilicity"],
                        color_name=["red", "pink", "green", "green"])

# filtered_df = dataframe_explorer(info_data[info_data["Source"]=="SI"], case=False)
filtered_cr = dataframe_explorer(CR_data, case=False)
output = st.empty()

# hide_dataframe_row_index  = """
#     <style>
#         button[data-testid="stDownloadButton"] {
#             display: none;
#         }
#     </style>
# """
# st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
# st.dataframe(filtered_df, use_container_width=True)


gb = GridOptionsBuilder.from_dataframe(filtered_cr)
gb.configure_grid_options(domLayout='normal')
gridOptions = gb.build()
AgGrid(filtered_cr, gridOptions=gridOptions, enable_enterprise_modules=False)

results_cr=len(filtered_cr)
st.markdown(f"There are :red[{results_cr}] MOFs found!")

with stdout(output.code, terminator=""):
    stoggle(
                "The meaning of each column in the dataframe",
                '''
                    🥷 coreid: CORE MOF ID, year + metal + topology + dimension + extension + No.<br>
                    🥷 refcode: CSD Refcode or SI file name + Extension + Charge<br>
                    🥷 name: common name<br>
                    🥷 mofid-v1: <a href="https://pubs.acs.org/doi/full/10.1021/acs.cgd.9b01050"> MOFid 1.0</a><br>
                    🥷 mofid-v2: <a href="https://github.com/snurr-group/mofid/tree/master/mofid-v2"> MOFid 2.0</a><br>
                    🥷 LCD (Å): pore-limiting diameter <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 PLD (Å): largest cavity diameter <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 LFPD (Å): largest free pore diameter <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 Density (g/cm3): crystal density <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 ASA (A2), ASA (m2/cm3) and ASA (m2/g): accessible surface area <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 NASA (A2), NASA (m2/cm3) and NASA (m2/g): non-accessible surface area <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 PV (A3) and PV (cm3/g): pore volume <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 VF: void fraction <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 NAV (A3) and NPV (cm3/g): non-accessible pore volume <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 NAV_VF: non-accessible void fraction <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 structure_dimension: dimensionality of each framework <a href="https://www.zeoplusplus.org/"> -> Zeo++</a><br>
                    🥷 topology(SingleNodes): topology defined by single node <a href="https://scipost.org/10.21468/SciPostChem.1.2.005"> -> CrystalNets</a><br>
                    🥷 topology(SingleNodes)-zeo: topology defined by single node from the <a href="https://www.iza-structure.org/databases/">IZA-SC database</a><br>
                    🥷 topology(AllNodes): topology defined by all nodes <a href="https://scipost.org/10.21468/SciPostChem.1.2.005"> -> CrystalNets</a><br>
                    🥷 topology(AllNodes)-zeo: topology defined by all nodes from the <a href="https://www.iza-structure.org/databases/">IZA-SC database</a><br>
                    🥷 catenation: number of nets <a href="https://scipost.org/10.21468/SciPostChem.1.2.005"> -> CrystalNets</a><br>
                    🥷 dimension_by_topo: dimension determined <a href="https://scipost.org/10.21468/SciPostChem.1.2.005"> -> CrystalNets</a><br>
                    🥷 hall: space group<br>
                    🥷 number_spacegroup: space group by number<br>
                    🥷 Metal Types: metal in the framework<br>
                    🥷 Has OMS: contains open metal site or not<br>
                    🥷 OMS Types: which metal is OMS<br>
                    🥷 Charge: <a href="https://pubs.acs.org/doi/10.1021/acs.jctc.4c00434">Charge model</a><br>
                    🥷 average_atomic_mass: average atomic mass of structure<br>
                    🥷 Heat_capacity@300K (J/g/K): <a href="https://www.nature.com/articles/s41563-022-01374-3">machine learning predicted heat capacity at 300 K</a><br>
                    🥷 std @ 300 K (J/g/K): <a href="https://www.nature.com/articles/s41563-022-01374-3">uncertainty of ML predicted heat capacity at 300 K</a><br>
                    🥷 Heat_capacity@350K (J/g/K): <a href="https://www.nature.com/articles/s41563-022-01374-3">machine learning predicted heat capacity at 350 K</a><br>
                    🥷 std @ 350 K (J/g/K): <a href="https://www.nature.com/articles/s41563-022-01374-3">uncertainty of ML predicted heat capacity at 350 K</a><br>
                    🥷 Heat_capacity@400K (J/g/K): <a href="https://www.nature.com/articles/s41563-022-01374-3">machine learning predicted heat capacity at 400 K</a><br>
                    🥷 std @ 400 K (J/g/K): <a href="https://www.nature.com/articles/s41563-022-01374-3">uncertainty of ML predicted heat capacity at 400 K</a><br>
                    🥷 k_cp (J/g/K/K): slope parameter, heat capacity = k_cp * temperature + cp0<br>
                    🥷 cp0 (J/g/K): intercept parameter, heat capacity = k_cp * temperature + cp0<br>
                    🥷 Pearson product-moment correlation coefficients: PPMCC of heat capacity predicted by ML with fitted parameters<br>
                    🥷 natoms: number of atoms in unit cell<br>
                    🥷 DOI: Digital Object Identifier of the paper reported structure<br>
                    🥷 Year: publication year<br>
                    🥷 Time: publication time<br>
                    🥷 Publication: publisher<br>
                    🥷 Extension: classification of structure after curation<br>
                    🥷 unmodified: whether the structure is unmodified compared with the original<br>
                    🥷 Thermal_stability (℃): ML predicted decomposition temperature<br>
                    🥷 Solvent_stability: ML predicted probability of solvent stability (stable if >0.5)<br>
                    🥷 Water_stability: ML predicted probability of water stability (stable if >0.5)<br>
                    🥷 KH_water: classification of hydrophilicity/hydrophobicity MOFs by Gibbs Ensemble Monte Carlo (GEMC) Simulation<br>
                '''
            )
    # ▶ Source: obtained source (CSD or SI)<br>

st.text("")
st.text("")
st.text("")
st.text("")
st.text("")

colored_header(
                label="Search from NCR Dataset",
                description="by KEY or INFO",
                color_name="blue-70",
                )

st.markdown('''#### :rainbow[by KEY]''')

NCR_data = pd.read_csv("./data/internal/All_data_20241205_search_all_ncr.csv")

def search_info_ncr(searchterm: str) -> list:
    if not searchterm:
        return []
    NCR_key = NCR_data[NCR_data.apply(lambda row: searchterm.lower() in row.astype(str).str.lower().to_string(), axis=1)]
    return NCR_key.to_dict(orient="records")


searchterm_ncr = st.text_input("Search in CoRE MOF NCR database...", key="search_df_key_ncr")

if searchterm_ncr:
    search_key_results = search_info_ncr(searchterm_ncr)
    n_results_ncr = len(search_key_results)
    
    st.markdown(f"There are :red[{n_results_ncr}] MOFs found!")
    if n_results_ncr > 0:
        selected_refcode = st.selectbox(
            "Select a MOF by REFCODE:",
            [row["refcode"] for row in search_key_results]
        )
        selected_row = next((row for row in search_key_results if row["refcode"] == selected_refcode), None)
        if selected_row:
            try:
                data0_ = NCR_data[NCR_data["refcode"] == selected_row["refcode"]]
                st.json(data0_.to_dict(orient="records"))
                if selected_row["DOI"] != "unknown":
                    st.markdown(f"[Read Original Paper](https://doi.org/{selected_row['DOI']})", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error visualizing MOF: {e}")
else:
    pass


if st_toggle_switch:

    st.markdown("**Supported Filters**")

    tagger_component("Type:", ["Extension for NCR"])
    tagger_component("NCR Cases by mofchecker:", ["has_lone_molecule", "has_atomic_overlaps",
                                                  "has_overcoordinated_c", "has_overcoordinated_n",
                                                  "has_overcoordinated_h", "has_undercoordinated_c",
                                                  "has_undercoordinated_n", "has_undercoordinated_rare_earth",
                                                  "has_undercoordinated_alkali_alkaline", "has_suspicious_terminal_oxo",
                                                  "has_geometrically_exposed_metal",  "no_carbon"],
                    color_name=["lightblue", "lightblue"
                                , "lightblue", "lightblue"
                                , "lightblue", "lightblue"
                                , "lightblue", "lightblue"
                                , "lightblue", "lightblue"
                                , "lightblue", "lightblue"])
    tagger_component("NCR Cases by Chen_Manz:", ["Isolated",
                                                 "Atom overlapping",
                                                 "Under bonded carbon",
                                                 "Over bonded carbon"],
                    color_name=["orange", "orange", "orange", "orange"])


st.markdown('''#### :rainbow[by INFO]''')

filtered_ncr = dataframe_explorer(NCR_data, case=False)

output = st.empty()


gb = GridOptionsBuilder.from_dataframe(filtered_ncr)
gb.configure_grid_options(domLayout='normal')
gridOptions = gb.build()
AgGrid(filtered_ncr, gridOptions=gridOptions, enable_enterprise_modules=False)

results_ncr=len(filtered_ncr)
st.markdown(f"There are :red[{results_ncr}] MOFs found!")

with stdout(output.code, terminator=""):
    stoggle(
                "The meaning of each column of mofchecker:",
                '''
                    🥷 has_lone_molecule:  The method determines if there are floating molecules by extracting subgraphs from the structure graph and checking their periodicity.<br>
                    🥷 has_atomic_overlaps: Overlapping atoms are identified if the minimum distance between two neighboring atoms is less than the sum of their covalent radii.<br>
                    🥷 has_overcoordinated_c / has_overcoordinated_n / has_overcoordinated_h: The method classifies an “overcoordinated” structure if the number of connected neighbors exceeds the specified quantity (H:1; N or C:4, ignoring connections to metals).<br>
                    🥷 has_undercoordinated_c / has_undercoordinated_n / has_undercoordinated_rare_earth / has_undercoordinated_alkali_alkaline: pore-limiting diameter: For alkali metals, alkaline earth metals, and rare earth metals, if the number of connected atoms is less than 4. For carbon, if the number of connected atoms is fewer than 3, the criterion is that the two atoms connected to the carbon and the carbon itself should not be aligned (with a tolerance of 10°). If any of the following three conditions is true, it is defined as an undercoordinated nitrogen: (a) if the number of connected atoms is fewer than 4, it must be checked that if the neighbors are only carbon atoms, then those carbon atoms should have more than 2 neighbors (i.e., it is not a CN group); (b) The nitrogen atom is connected to two other atoms and is not linear (not 𝑠𝑝 hybridized), and the minimum value of all dihedral angles related to this nitrogen atom is neither close to 0° nor 180° degrees (with a tolerance of 25°), or any bond length is greater than 1.4 Å; (c) The smallest angle formed by the three atoms bonded to the nitrogen is less than 110° (with a tolerance of 10°), and among the neighboring atoms, there is one metal atom and two hydrogen atoms.<br>
                    🥷 has_suspicious_terminal_oxo: An oxygen atom is considered part of an oxo group if it is connected to a metal and not bonded to any other atoms (due to missing H atoms on "terminal" metal-oxo species that should be OH groups or H2O groups).<br>
                    🥷 has_geometrically_exposed_metal: : Alkali metals, alkaline earth metals, and rare earth metals are considered geometrically exposed if they form bonds with their neighbors at angles greater than 150º and have fewer than six connected neighbors.<br>
                '''
            )

with stdout(output.code, terminator=""):
    stoggle(
                "The meaning of each column of Chen_Manz:",
                '''
                    🥷 Isolated: Has atoms do no connect with framework.<br>
                    🥷 Atom overlapping: If the distance between them is less than half the sum of their radii.<br>
                    🥷 Under bonded carbon: If the sum is greater than or equal to 5.5.<br>
                    🥷 Over bonded carbon:  If the sum of its bond orders is less than 3.3.<br>
                '''
            )
    
st.divider()