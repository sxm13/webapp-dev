import streamlit as st
from stmol import *
import pandas as pd

from streamlit_extras.markdownlit import mdlit
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

from io import StringIO, BytesIO
import os, warnings
import time
import schedule
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Calculation", page_icon="./figures/logo.png")
logo_url = "./figures/logo.png"
st.sidebar.image(logo_url)

st.logo(logo_url, link="https://zenodo.org/communities/core-mofs/records?q=&l=list&p=1&s=10&sort=newest")

with st.sidebar:
    st.markdown("© 2024 CoRE MOF Project. CC-BY-4.0 License.")

st.title("Calculation")

st.divider()

mdlit(
        """**Topology** -> [CrystalNets](https://progs.coudert.name/topology), 
        """
    )
mdlit(
        """**MOFid** -> [MOFid-v1](https://snurr-group.github.io/web-mofid/sbu.html).
        """
    )
mdlit(
        """**Thermal and activation stability prediction** -> [mofsimplify](https://mofsimplify.mit.edu/).
        """
    )

uploaded_file = st.file_uploader("Upload your CIF file for Property Calculation :art: ", type=["cif"])

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "use_example" not in st.session_state:
    st.session_state.use_example = False

if st.button("Example CIF (CSD REFCODE: DOTSOV01; Common Name: Cu-BTC)",icon="🚨", use_container_width=True):
    st.session_state.use_example = True
    st.session_state.uploaded_file = "./data/example/DOTSOV01_ASR.cif"

from pathlib import Path


if uploaded_file is not None:
    st.session_state.use_example = False
    st.session_state.uploaded_file = uploaded_file

from utils.calc import others

if st.session_state.uploaded_file:
    if isinstance(st.session_state.uploaded_file, str):
        file_path = Path(st.session_state.uploaded_file)
        file_name = file_path.stem
        with file_path.open("r", encoding="utf-8") as f:
            _cif = f.read()
        uploaded_file_path = "./data/example/DOTSOV01_ASR.cif"
        sg = others.SpaceGroup(uploaded_file_path)
        m = others.Mass(uploaded_file_path)
        v = others.Volume(_cif)
        na = others.n_atom(uploaded_file_path)
        
    else:
        file_name = Path(st.session_state.uploaded_file.name).stem
        _cif = st.session_state.uploaded_file.getvalue().decode("utf-8")
        uploaded_file_path = "./data/tmp_files/"+file_name+".cif"
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        sg = others.SpaceGroup(StringIO(_cif))
        m = others.Mass(StringIO(_cif))
        v = others.Volume(_cif)
        na = others.n_atom(StringIO(_cif))
    col = st.columns((2, 2), gap='medium')
    with col[0]:
  
        col1, col2 = st.columns(2)
        with col1:
            bcolor = st.color_picker('Background Color','#89cff0')
        with col2:
            style = st.selectbox('style',['stick','sphere','cartoon','line','cross'])
        xyzview = py3Dmol.view(width=300, height=250)
        xyzview.addModel(_cif, "cif")
        xyzview.setStyle({style:{'color':'spectrum'}})
        xyzview.setBackgroundColor(bcolor)
        xyzview.zoomTo()
        showmol(xyzview, width=300, height=250)
        
        add_vertical_space(1)

    with col[1]:
        

        add_vertical_space(5)
        

        basic_data = pd.DataFrame({
                                  "Space Group (Number)": [f"{sg['hall_symbol']} ({sg['space_group_number']})"],
                                  "Number of Atoms": [na["number_atoms"]],
                                  "Mass / " + m["unit"]: [round(m["total_mass"], 4)],
                                  "Volume / " + v["unit"]: [round(v["total_volume"], 4)]
                              })

    
        unit = m["unit"] if "unit" in m else "unit"
        basic_info = pd.DataFrame({
                                    "Parameter": ["Space Group (Number)", "Number of Atoms", f"Mass / {unit}", "Volume / Å³"],
                                    "Value": [
                                        basic_data.iloc[0, 0],
                                        basic_data.iloc[0, 1],
                                        basic_data.iloc[0, 2],
                                        basic_data.iloc[0, 3]
                                    ]
                                })
        st.dataframe(basic_info, hide_index=True)

    colored_header(
                label="MOF Analysis",
                description="",
                color_name="green-50",
              )
    

    tabl, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Zeo++ (Pore, Dim)','Open Metal Site', 'Revised Autocorrelation', 'Heat Capacity', 'PACMAN charge', 'Water Stability'])
  
    with tabl:
        from utils.calc import pore, dim
        st.markdown("""T.F. Willems, C.H. Rycroft, M. Kazi, J.C. Meza, and M. Haranczyk.
                    Algorithms and tools for high-throughput geometry- based analysis of crystalline porous materials. 
                    Microporous and Mesoporous Materials. 149, 134-141 (2012). 
                    https://doi.org/10.1016/j.micromeso.2011.08.020.""")
        cols = st.columns((4, 4), gap='medium')

        with cols[0]:
            st.markdown("**Pore Diameter**")
            ha_pd = st.radio(
                                "High Accuracy",
                                [True, False],
                                index=0,
                                key="high_accuracy_pd"
                            )

            run_pd = st.button("Calculate", key="run_pd")
            
            if run_pd:
                with st.spinner('Calculating...'):
                    try:
                        # with open(uploaded_file_path, "wb") as f:
                        #     f.write(uploaded_file.getbuffer())

                        results_pd = pore.PoreDiameter(uploaded_file_path, ha_pd)

                        PLD = round(results_pd["PLD"], 4)
                        LCD = round(results_pd["LCD"], 4)
                        LFPD = round(results_pd["LFPD"], 4)

                        pd_info = pd.DataFrame({
                                                    "Parameter": ["PLD / Å", "LCD / Å", "LFPD / Å"],
                                                    "Value": [
                                                        PLD,
                                                        LCD,
                                                        LFPD
                                                    ]
                                                })
                        st.dataframe(pd_info, hide_index=True)
                
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        
        add_vertical_space(1)
        st.divider()

        with cols[1]:
            st.markdown("**Dimension**")
            ha_dim = st.radio(
                                "High Accuracy",
                                [True, False],
                                index=0,
                                key="high_accuracy_dim"
                            )
            probe_radius_dim = st.number_input("probe radiu", min_value=0.000, max_value=3.000, value=1.655, step=0.001, key="probe_dim")

            run_dim = st.button("Calculate", key="run_dim")
            
            if run_dim:
                with st.spinner('Calculating...'):
                    # try:
                        # with open(uploaded_file_path, "wb") as f:
                        #     f.write(uploaded_file.getbuffer())

                        results_chan = dim.ChanDim(uploaded_file_path, probe_radius_dim, ha_dim)
                        results_strinfo = dim.FrameworkDim(uploaded_file_path, ha_dim)

                        chan_dim = results_chan["Dimention"]
                        fram_dim = results_strinfo["Dimention"]                      

                        dim_info = pd.DataFrame({
                                                "Parameter": ["Dimension of Channel", "Dimension of Framework"],
                                                "Value": [
                                                    chan_dim,
                                                    fram_dim
                                                    ]
                                                })
                        st.dataframe(dim_info, hide_index=True)

                    # except Exception as e:
                    #     st.error(f"An error occurred: {e}")

        add_vertical_space(1)
        

        colss = st.columns((4, 4), gap='medium')

        with colss[0]:
           
            st.markdown("**Pore Volume**")
            ha_pv = st.radio(
                                "High Accuracy",
                                [True, False],
                                index=0,
                                key="high_accuracy_pv"
                            )
            chan_radius_pv = st.number_input("chan radiu", min_value=0.000, max_value=3.000, value=0.000, step=0.001, key="chan_pv")
            probe_radius_pv = st.number_input("probe radiu", min_value=0.000, max_value=3.000, value=0.000, step=0.001, key="probe_pv")
            num_samples_pv = st.number_input("number of samples", min_value=1, value=5000, step=500, key="n_samples_pv")

            run_pv = st.button("Calculate", key="run_pv")
            
            if run_pv:
                with st.spinner('Calculating...'):
                    try:
                        with open(uploaded_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        results_pv = pore.PoreVolume(uploaded_file_path, chan_radius_pv, probe_radius_pv, num_samples_pv, ha_pv)

                        PV1 = round(results_pv["PV"][0], 4)
                        PV2 = round(results_pv["PV"][1], 4)

                        NPV1 = round(results_pv["NPV"][0], 4)
                        NPV2 = round(results_pv["NPV"][1], 4)

                        VF = round(results_pv["VF"], 4)
                        NVF = round(results_pv["NVF"], 4)

                        pv_info = pd.DataFrame({
                                                    "Parameter": ["PV / Å³", "PV /cm³ g⁻¹ ",
                                                              "NPV / Å³", "NPV / cm³ g⁻¹ ",
                                                              "VF", "NVF"],
                                                    "Value": [
                                                        PV1,
                                                        PV2,
                                                        NPV1,
                                                        NPV2,
                                                        VF,
                                                        NVF
                                                        ]
                                                    })
                        st.dataframe(pv_info, hide_index=True)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        with colss[1]:
          st.markdown("**Surface Area**")
          ha_sa = st.radio(
                              "High Accuracy",
                              [True, False],
                              index=0,
                              key="high_accuracy_sa"
                          )
          chan_radius_sa = st.number_input("chan radiu", min_value=0.000, max_value=3.000, value=1.655, step=0.001, key="chan_sa")
          probe_radius_sa = st.number_input("probe radiu", min_value=0.000, max_value=3.000, value=1.655, step=0.001, key="probe_sa")
          num_samples_sa = st.number_input("number of samples", min_value=1, value=5000, step=500, key="n_samples_sa")

          run_sa = st.button("Calculate", key="run_sa")
          
          if run_sa:
                with st.spinner('Calculating...'):
                    try:
                        # with open(uploaded_file_path, "wb") as f:
                        #     f.write(uploaded_file.getbuffer())

                        results_sa = pore.SurfaceArea(uploaded_file_path, chan_radius_sa, probe_radius_sa, num_samples_sa, ha_sa)

                        ASA1 = round(results_sa["ASA"][0], 0)
                        ASA2 = round(results_sa["ASA"][1], 0)
                        ASA3 = round(results_sa["ASA"][2], 0)

                        NASA1 = round(results_sa["NASA"][0], 0)
                        NASA2 = round(results_sa["NASA"][1], 0)
                        NASA3 = round(results_sa["NASA"][2], 0)

                        sa_info = pd.DataFrame({
                                                "Parameter": ["ASA / Å²", "ASA / m² cm⁻³",
                                                          "ASA / m² g⁻¹", "NASA / Å²",
                                                          "NASA / m² cm⁻³", "NASA / m² g⁻¹"],
                                                "Value": [
                                                    ASA1,
                                                    ASA2,
                                                    ASA3,
                                                    NASA1,
                                                    NASA2,
                                                    NASA3
                                                    ]
                                                })
                        st.dataframe(sa_info, hide_index=True)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")


    with tab2:
        from utils.calc import oms
        st.markdown("""Yongchul G. Chung, Emmanuel Haldoupis, Benjamin J. Bucior, Maciej Haranczyk, Seulchan Lee, Hongda Zhang, Konstantinos D. Vogiatzis, Marija Milisavljevic, Sanliang Ling, Jeffrey S. Camp, Ben Slater, J. Ilja Siepmann, David S. Sholl, and Randall Q. Snurr
                        Journal of Chemical & Engineering Data 2019 64 (12), 5985-5998.
                        https://doi.org/10.1021/acs.jced.9b00835""")
        # with open(uploaded_file_path, "wb") as f:
        #     f.write(uploaded_file.getbuffer())

        run_oms = st.button("Calculate", key="run_oms") 
        if run_oms:
          with st.spinner('Calculating...'):
            oms_result=oms.get_from_file(uploaded_file_path)

            metal = oms_result["Metal Types"]
            has_oms = oms_result["Has OMS"]
            oms_type = oms_result["OMS Types"]

            oms_info = pd.DataFrame({
                                    "Parameter": ["Metal Types", "Has OMS", "OMS Types"],
                                    "Value": [
                                        metal,
                                        has_oms,
                                        oms_type
                                    ]
                                })
            st.dataframe(oms_info, hide_index=True)


    with tab3:
        from utils.calc import RAC
        st.markdown("""Jon Paul Janet and Heather J. Kulik. 
                    Resolving Transition Metal Chemical Space: Feature Selection for Machine Learning and Structure-Property Relationships.
                    The Journal of Physical Chemistry A. 121 (46), 8939-8954 (2017).
                    https://pubs.acs.org/doi/10.1021/acs.jpca.7b08750""")
        run_rac = st.button("Calculate", key="run_rac") 
        if run_rac:
            with st.spinner('Calculating...'):
                
                # with open(uploaded_file_path, "wb") as f:
                #     f.write(uploaded_file.getbuffer())

                result_rac = RAC.get(uploaded_file_path)
                
                from utils.calc.RAC import metal_fnames, linker_fnames, fg_fnames

                metal_dfs = []
                linker_dfs = []
                fg_dfs = []

                for metal in metal_fnames:
                    metal_dfs.append(pd.DataFrame({metal: [result_rac["Metal"][metal]]}))
                    
                st.markdown("""Metal""")
                metal_ = pd.concat(metal_dfs, axis=1)
                st.dataframe(metal_, hide_index=True)
                
                for linker in linker_fnames:
                    linker_dfs.append(pd.DataFrame({linker: [result_rac["Linker"][linker]]}))
                linker_ = pd.concat(linker_dfs, axis=1)
                st.markdown("""Linker""")
                st.dataframe(linker_, hide_index=True)
            
                for fg in fg_fnames:
                    fg_dfs.append(pd.DataFrame({fg: [result_rac["Function-group"][fg]]}))
                fg_ = pd.concat(fg_dfs, axis=1)
                st.markdown("""Functional-group""")
                st.dataframe(fg_, hide_index=True)

                X_metal = metal_[metal_fnames].to_numpy()
                X_linker = linker_[linker_fnames].to_numpy()
                X_fg = fg_[fg_fnames].to_numpy()

                import plotly.graph_objects as go
                from sklearn.manifold import TSNE

                df = pd.read_csv('./data/internal/features_ASR.csv')

                X_metal_df = pd.DataFrame(X_metal, columns=metal_fnames)
                X_metal_df["highlight"] = True 
                df_metal = df[metal_fnames].copy()
                df_metal["highlight"] = False

                combined_metal_df = pd.concat([df_metal, X_metal_df], ignore_index=True)

                n_samples_metal = combined_metal_df.shape[0]
                metal_perplexity = min(30, max(1, n_samples_metal // 2))
                tsne = TSNE(n_components=2, perplexity=metal_perplexity)
                tsne_result = tsne.fit_transform(combined_metal_df[metal_fnames].dropna().values)

                tsne_df = pd.DataFrame(tsne_result, columns=[f"dim {i + 1}" for i in range(2)])
                tsne_df["highlight"] = combined_metal_df["highlight"]
                traces = []

                normal_points = tsne_df[~tsne_df["highlight"]]
                traces.append(go.Scatter(
                    x=normal_points["dim 1"],
                    y=normal_points["dim 2"],
                    mode="markers",
                    name="CoRE MOF 2024 ASR Dataset",
                    marker=dict(size=6, color="blue")
                ))

                highlight_points = tsne_df[tsne_df["highlight"]]
                traces.append(go.Scatter(
                    x=highlight_points["dim 1"],
                    y=highlight_points["dim 2"],
                    mode="markers",
                    marker_symbol="star",
                    marker=dict(size=10, color="red"),
                    name="Your Structure"
                ))

                fig_tsne = go.Figure(data=traces)
                fig_tsne.update_layout(
                    title=f"T-SNE for Metal: (2D)",
                    xaxis_title="dim 1",
                    yaxis_title="dim 2"
                )
                st.plotly_chart(fig_tsne, use_container_width=True)


                X_linker_df = pd.DataFrame(X_linker, columns=linker_fnames)
                X_linker_df["highlight"] = True
                df_linker = df[linker_fnames].copy()
                df_linker["highlight"] = False

                combined_linker_df = pd.concat([df_linker, X_linker_df], ignore_index=True)

                n_samples_linker = combined_linker_df.shape[0]
                linker_perplexity = min(30, max(1, n_samples_linker // 2))
                tsne = TSNE(n_components=2, perplexity=linker_perplexity)
                tsne_result = tsne.fit_transform(combined_linker_df[linker_fnames].dropna().values)

                tsne_df = pd.DataFrame(tsne_result, columns=[f"dim {i + 1}" for i in range(2)])
                tsne_df["highlight"] = combined_linker_df["highlight"]
                traces = []

                normal_points = tsne_df[~tsne_df["highlight"]]
                traces.append(go.Scatter(
                    x=normal_points["dim 1"],
                    y=normal_points["dim 2"],
                    mode="markers",
                    name="CoRE MOF 2024 ASR Dataset",
                    marker=dict(size=6, color="green")
                ))

                highlight_points = tsne_df[tsne_df["highlight"]]
                traces.append(go.Scatter(
                    x=highlight_points["dim 1"],
                    y=highlight_points["dim 2"],
                    mode="markers",
                    marker_symbol="star",
                    marker=dict(size=10, color="red"),
                    name="Your Structure"
                ))

                fig_tsne = go.Figure(data=traces)
                fig_tsne.update_layout(
                    title=f"T-SNE for Linker: (2D)",
                    xaxis_title="dim 1",
                    yaxis_title="dim 2"
                )
                st.plotly_chart(fig_tsne, use_container_width=True)

                X_fg_df = pd.DataFrame(X_fg, columns=fg_fnames)
                X_fg_df["highlight"] = True
                df_fg = df[fg_fnames].copy()
                df_fg["highlight"] = False

                combined_fg_df = pd.concat([df_fg, X_fg_df], ignore_index=True)

                n_samples_fg = combined_fg_df.shape[0]
                fg_perplexity = min(30, max(1, n_samples_fg // 2))
                tsne = TSNE(n_components=2, perplexity=fg_perplexity)
                tsne_result = tsne.fit_transform(combined_fg_df[fg_fnames].dropna().values)

                tsne_df = pd.DataFrame(tsne_result, columns=[f"dim {i + 1}" for i in range(2)])
                tsne_df["highlight"] = combined_fg_df["highlight"]
                traces = []

                normal_points = tsne_df[~tsne_df["highlight"]]
                traces.append(go.Scatter(
                    x=normal_points["dim 1"],
                    y=normal_points["dim 2"],
                    mode="markers",
                    name="CoRE MOF 2024 ASR Dataset",
                    marker=dict(size=6, color="grey")
                ))

                highlight_points = tsne_df[tsne_df["highlight"]]
                traces.append(go.Scatter(
                    x=highlight_points["dim 1"],
                    y=highlight_points["dim 2"],
                    mode="markers",
                    marker_symbol="star",
                    marker=dict(size=10, color="red"),
                    name="Your Structure"
                ))

                fig_tsne = go.Figure(data=traces)
                fig_tsne.update_layout(
                    title=f"T-SNE for Functional-group: (2D)",
                    xaxis_title="dim 1",
                    yaxis_title="dim 2"
                )
                st.plotly_chart(fig_tsne, use_container_width=True)


    with tab4:
        from utils.pred import cp
        st.markdown("""S. M. Moosavi, B. Á. Novotny, D. Ongari, E. Moubarak, M. Asgari, Ö. Kadioglu, et al.
                        A data-science approach to predict the heat capacity of nanoporous materials. Nat. Mater. 21, 1419–1425 (2022).
                        https://doi.org/10.1038/s41563-022-01374-3""")
        T = st.multiselect(
                        "Temperature",
                        [300, 350, 400],
                        [300],
                    )
        run_cp = st.button("Predict", key="run_cp")
        if run_cp:     
            with st.spinner('Predicting...'):

                # with open(uploaded_file_path, "wb") as f:
                #     f.write(uploaded_file.getbuffer())
                
                result_cp=cp.run(uploaded_file_path, file_name, T=T)

                cp_info_list = []
                for t in T:
                    cp_g = round(result_cp[str(t)+"_mean"][0], 4)
                    cp_m = round(result_cp[str(t)+"_mean"][1], 4)
                    cp_g_std = round(result_cp[str(t)+"_std"][0], 4)
                    cp_m_std = round(result_cp[str(t)+"_std"][1], 4)

                    # st.markdown("""
                    #   <style>
                    #   .custom-table {
                    #       border-collapse: collapse;
                    #       width: 100%;
                    #       margin-top: 20px;
                    #   }
                    #   .custom-table th, .custom-table td {
                    #       border: 1px solid #ddd;
                    #       padding: 8px;
                    #       text-align: left;
                    #   }
                    #   .custom-table th {
                    #       background-color: #f2f2f2;
                    #       font-weight: bold;
                    #   }
                    #   .custom-table tr:hover {
                    #       background-color: #f1f1f1;
                    #   }
                    #   </style>
                    #   """, unsafe_allow_html=True)
                    # st.markdown(f"""
                    # <table class="custom-table">
                    #     <thead>
                    #         <tr>
                    #             <th>T / K</th>
                    #             <th>cp / J g<sup>-1</sup> K<sup>-1</sup></th>
                    #             <th>cp std / J g<sup>-1</sup> K<sup>-1</sup></th>
                    #             <th>cp / J mol<sup>-1</sup> K<sup>-1</sup></th>
                    #             <th>cp std / J mol^-1 K^-1</th>
                    #         </tr>
                    #     </thead>
                    #     <tbody>
                    #         <tr>
                    #             <td>{t}</td>
                    #             <td>{cp_g}</td>
                    #             <td>{cp_g_std}</td>
                    #             <td>{cp_m}</td>
                    #             <td>{cp_m_std}</td>
                    #         </tr>
                    #     </tbody>
                    # </table>
                    # """, unsafe_allow_html=True)

                    cp_single_info = pd.DataFrame({
                                            "info": ["T / K",
                                                    "Cₚ / J g⁻¹ K⁻¹",
                                                        "Cₚ / J mol⁻¹ K⁻¹",
                                                        "Cₚ std / J g⁻¹ K⁻¹",
                                                        "Cₚ std / J mol⁻¹ K⁻¹"],
                                            "Value": [
                                                t,
                                                cp_g,
                                                cp_m,
                                                cp_g_std,
                                                cp_m_std
                                                ]
                                            })
                    st.dataframe(cp_single_info, hide_index=True)
    with tab5:
        from PACMANCharge import pmcharge
        st.markdown("""Guobin Zhao and Yongchul G. Chung.
                        PACMAN: A Robust Partial Atomic Charge Predicter for Nanoporous Materials Based on Crystal Graph Convolution Networks. Journal of Chemical Theory and Computation 20 (12), 5368-5380 (2024).
                        https://doi.org/10.1021/acs.jctc.4c00434""")
        charge_type = st.radio(
                        "Charge Type",
                        ["DDEC6", "Bader", "CM5", "REPEAT"],
                        index=0,
                        key="charge model"
                        )
        atom_type = st.radio(
                        "Atom Type",
                        [True, False],
                        index=0,
                        key="keep atom type"
                        )
        neutral = st.radio(
                        "Neutral",
                        [True, False],
                        index=0,
                        key="keep zero"
                        )
        bond = st.radio(
                        "Keep Connect",
                        [True, False],
                        index=0,
                        key="keep bond"
                        )
        digits = st.number_input("Digits", min_value=1, value=6, max_value=15)
        run_pacman = st.button("Predict", key="run_pacman")
        if run_pacman:
            with st.spinner('Predicting...'):
                pmcharge.predict(cif_file=uploaded_file_path,
                                    charge_type=charge_type,
                                    digits=digits,
                                    atom_type=atom_type,
                                    neutral=neutral,
                                    keep_connect=bond)
                
                pacman_file_path = uploaded_file_path.replace(".cif", "_pacman.cif")
                with open(pacman_file_path, "r", encoding="utf-8") as f:
                    pacman_file_content = f.read()
                st.download_button(label="📍 Download structure with PACMAN charge",
                                data=pacman_file_content,
                                file_name=pacman_file_path.split("/")[-1],
                                mime='text/plain')
    with tab6:
        from utils.pred import water
        st.markdown("""Gianmarco G. Terrones, Shih-Peng Huang, Matthew P. Rivera, Shuwen Yue, Alondra Hernandez, and Heather J. Kulik.
                        Metal–Organic Framework Stability in Water and Harsh Environments from Data-Driven Models Trained on the Diverse WS24 Data Set. Journal of the American Chemical Society. 146 (29), 20333-20348 (2024).
                        https://doi.org/10.1021/jacs.4c05879""")
        
        run_ws = st.button("Predict", key="run_ws_1")
        if run_ws:     
            with st.spinner('Predicting...'):
                
                result_ws=water.run(uploaded_file_path)
                ws_data = round(result_ws["water probability"], 4)
        
                ws_info = pd.DataFrame({
                                        "info": ["Water Stability"],
                                        "Value": [ws_data]
                                        })
                st.dataframe(ws_info, hide_index=True)

def delete_old_files(folder_path, age_limit_seconds):

    current_time = time.time()

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):

            creation_time = os.path.getctime(file_path)
            if current_time - creation_time > age_limit_seconds:
                os.remove(file_path)

folder_path = "./data/tmp_files/"
age_limit_seconds = 3600

schedule.every(1).minutes.do(delete_old_files, folder_path, age_limit_seconds)

while True:
    schedule.run_pending()
    time.sleep(1)
