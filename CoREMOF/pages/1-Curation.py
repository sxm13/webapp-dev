import streamlit as st
from stmol import *

import zipfile

from io import StringIO, BytesIO

from streamlit_extras.add_vertical_space import add_vertical_space
# from streamlit_extras.concurrency_limiter import concurrency_limiter
from streamlit_extras.customize_running import center_running

from ase.io import read, write
from ase.build import make_supercell
from pymatgen.io.ase import AseAtomsAdaptor
from streamlit_extras.colored_header import colored_header

import warnings
from data.info.atoms_definitions import METAL
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
warnings.filterwarnings("ignore")

from utils.clean import run_fsr, run_asr
from utils.check import run_check

st.set_page_config(page_title="Curation", page_icon="./figures/logo.png")
logo_url = "./figures/logo.png"
st.sidebar.image(logo_url)
st.logo(logo_url, link="https://zenodo.org/communities/core-mofs/records?q=&l=list&p=1&s=10&sort=newest")

st.title("Curation")

with st.sidebar:
    st.markdown("© 2024 CoRE MOF Project. CC-BY-4.0 License.")

st.divider()

uploaded_file = st.file_uploader("Upload your CIF file for Curation :art: ", type=["cif"])

# @concurrency_limiter(max_concurrency=3)
def vis(_cif, scaling_matrix=[[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]):
    viewer = py3Dmol.view(width=800, height=500)
    try:
        atoms = read(StringIO(_cif), format="cif")
        supercell = make_supercell(atoms, scaling_matrix)
        cif_io = BytesIO()
        write(cif_io, supercell, format="cif")
        cif_content = cif_io.getvalue().decode('utf-8')
        viewer.addModel(cif_content, "cif")
        viewer.setStyle({'stick': {}})
        viewer.addUnitCell()
        viewer.zoomTo()
        showmol(viewer, width=800, height=500)
        return cif_content
    except Exception as e:
        st.error(f"{e}")
        st.markdown('''**Please upload a correct structure.** :sob:''')
        
if "use_example" not in st.session_state:
    st.session_state.use_example = False

if st.button("Example CIF (CSD REFCODE: ABAVIJ)",icon="🚨", use_container_width=True):
    st.session_state.use_example = True

if st.session_state.use_example:
    uploaded_file = "./data/example/ABAVIJ.cif"

if uploaded_file is not None:
    if isinstance(uploaded_file, str):
        file_name = uploaded_file.split('/')[-1].split('.')[0]
        with open(uploaded_file, "r", encoding="utf-8") as f:
            _cif = f.read()
        bytes_data =_cif.encode("utf-8")
    else:
        file_name = uploaded_file.name.split('.')[0]
        _cif = uploaded_file.read().decode("utf-8")
        bytes_data = uploaded_file.getvalue()

    add_vertical_space(2)

    colored_header(
                label="Supercell",
                description="",
                color_name="red-50",
            )

    st.markdown("Make Supercell")
    col1, col2, col3 = st.columns(3)
    with col1:
      nx = st.number_input("a",min_value=1,max_value=5)
    with col2:
      ny = st.number_input("b",min_value=1,max_value=5)
    with col3:
      nz = st.number_input("c",min_value=1,max_value=5)

    with st.spinner('Wait for it...'):
      cif_content = vis(_cif, scaling_matrix=[[nx, 0, 0],[0, ny, 1],[0, 0, nz]])
      cif_sc = BytesIO(cif_content.encode("utf-8"))
      st.download_button(label="📍 Download supercell structure", data=cif_sc, file_name=f"{file_name}_supercell.cif", mime='text/plain')
    
    colored_header(
                label="Process Your Structure",
                description="",
                color_name="red-50",
            )

        
    click_curate = st.button("🏃‍♂️ Start Curating Your MOF")
    # @concurrency_limiter(max_concurrency=3)
    def check_structure(structure):
        has_metal = any(METAL.get(atom.symbol) for atom in structure)
        has_carbon = any(atom.symbol == 'C' for atom in structure)
        is_MOF = True
        if has_metal:
            pass
        else:
            is_MOF = False
        if has_carbon:
            pass
        else:
            is_MOF = False
        return is_MOF

    # @concurrency_limiter(max_concurrency=3)
    def make_p1(structure):
        sga = SpacegroupAnalyzer(structure)
        structure_p1 = sga.get_primitive_standard_structure(international_monoclinic=True, keep_site_properties=False)
        return structure_p1
    
    if click_curate:
        center_running()
        
        with st.status("Curating...", expanded=True) as status:

            st.write("1. Check .cif file...")
            # bytes_data = uploaded_file.getvalue()
            structures = read(BytesIO(bytes_data), format='cif',index=":")

            if len(structures) > 1:
                st.error("The \".cif\" file you uploaded contains multiple structures, please download them and upload a single structure.")
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for i, structure in enumerate(structures):
                        cif_file = BytesIO()
                        write(cif_file, structure, format='cif')
                        zip_file.writestr(f"structure_{i + 1}.cif", cif_file.read())
                zip_buffer.seek(0)
                st.download_button(
                                    label="Download splitted crystal files",
                                    data=zip_buffer,
                                    file_name="structures.zip",
                                    mime="application/zip"
                                    )

            else:
                if "preprocess_downloaded" not in st.session_state:
                    st.session_state["preprocess_downloaded"] = False
                
                if "fsr_downloaded" not in st.session_state:
                    st.session_state["fsr_downloaded"] = False
                if "asr_downloaded" not in st.session_state:
                    st.session_state["asr_downloaded"] = False

                st.write("2. Check atoms...")
                structure = structures[0]
                try:
                    if check_structure(structure):
                        st.success("😍 Your structure has carbon and metal")
                    else:
                        st.error("🙁 Your uploaded structure is missing either metal or carbon.")
                except:
                    st.error("🙁 Please upload a valid CIF file.")

                adaptor = AseAtomsAdaptor()

                st.write("3. Make primitive cell and P1...")
                structure_pri = adaptor.get_structure(structure)
                structure_pri_P1 = make_p1(structure_pri)
                st.success("✅ Make primitive cell and P1 sucessfully")
                cif_pre_ = structure_pri_P1.to(fmt="cif")
                cif_pre = BytesIO(cif_pre_.encode("utf-8"))
                # st.download_button(label="📍 Download Preprocessed Structure",
                #                     data=cif_pre, file_name=f"{file_name}_preprocess.cif",
                #                     mime='text/plain', key="download_preprocess",
                #                     on_click=lambda: st.session_state.update({"preprocess_downloaded": True}))
                
                if st.session_state["preprocess_downloaded"]:
                    st.success("🎉 Preprocess CIF has been downloaded!")

                struc = read(cif_pre, format='cif')

                st.write("4. Check NCR for original structure...")
                run_checker = run_check(struc)
                ori_c, ori_m = run_checker()
                if len(ori_c) == 0 and len(ori_m) == 0:
                    st.success("😍 Your Uploaded Structure is good!")
                else:
                    if len(ori_c) > 0:
                        st.warning("Problem of original structure by Chen_Manz: " + ", ".join(ori_c))
                    if len(ori_m) > 0:
                        st.warning("Problem of original structure by mofchecker: " + ", ".join(ori_m))

                # st.divider()

                st.write("5. Remove free solvent...")
                structure_fsr, has_ion_fsr = run_fsr(struc)

                st.write("6. Remove all solvent...")
                structure_asr, has_ion_asr = run_asr(struc)
                st.success("✅ Clean your structure sucessfully")
                if has_ion_fsr or has_ion_asr:
                    st.warning("Your structure has ion, please ignore free ion checking!")
                adaptor = AseAtomsAdaptor()
                
                cif_fsr_ = adaptor.get_structure(structure_fsr).to(fmt="cif")
                cif_fsr = BytesIO(cif_fsr_.encode("utf-8"))
                # st.markdown('''FSR Structure''')
                # vis(cif_fsr.read().decode("utf-8"))
                st.write("7. Check NCR for FSR...")
                run_checker = run_check(structure_fsr)
                fsr_c, fsr_m = run_checker()
                if len(fsr_c) == 0 and len(fsr_m) == 0:
                    st.success("😄 FSR structure is good!")
                else:
                    if len(fsr_c) > 0:
                        st.warning("🙁 Problem of FSR structure by Chen_Manz: " + ", ".join(fsr_c))
                    if len(fsr_m) > 0:
                        st.warning("🙁 Problem of FSR structure by mofchecker: " + ", ".join(fsr_m))
                # st.download_button(label="📍 Download FSR Structure", data=cif_fsr, file_name=f"{file_name}_FSR.cif",
                #                    mime='text/plain', key="download_FSR",
                #                    on_click=lambda: st.session_state.update({"fsr_downloaded": True}))

                if st.session_state["fsr_downloaded"]:
                    st.success("🎉 FSR CIF has been downloaded!")

                cif_asr_ = adaptor.get_structure(structure_asr).to(fmt="cif")
                cif_asr = BytesIO(cif_asr_.encode("utf-8"))
                
                # st.markdown('''ASR Structure''')
                # vis(cif_asr.read().decode("utf-8"))
                st.write("8. Check NCR for ASR...")
                run_checker = run_check(structure_asr)
                asr_c, asr_m = run_checker()
                if len(asr_c) == 0 and len(asr_m) == 0:
                    st.success("😄 ASR structure is good!")
                else:
                    if len(asr_c) > 0:
                        st.warning("🙁 Problem of ASR structure by Chen_Manz: " + ", ".join(asr_c))
                    if len(asr_m) > 0:
                        st.warning("🙁 Problem of ASR structure by mofchecker: " + ", ".join(asr_m))
                # st.download_button(label="📍 Download ASR Structure", data=cif_asr, file_name=f"{file_name}_ASR.cif",
                #                    mime='text/plain', key="download_ASR",
                #                    on_click=lambda: st.session_state.update({"asr_downloaded": True}))
                
                if st.session_state["asr_downloaded"]:
                    st.success("🎉 ASR CIF has been downloaded!")
                
                def create_zip_file(file_dict):
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for file_name, file_content in file_dict.items():
                            
                            if isinstance(file_content, BytesIO):
                                file_content.seek(0)
                                zip_file.writestr(file_name, file_content.read())
                            else:
                            
                                zip_file.writestr(file_name, file_content.encode('utf-8'))
                    zip_buffer.seek(0) 
                    return zip_buffer

                file_dict = {
                    f"{file_name}_preprocess.cif": cif_pre,  
                    f"{file_name}_FSR.cif": cif_fsr, 
                    f"{file_name}_ASR.cif": cif_asr,
                }

                zip_file = create_zip_file(file_dict)

                st.download_button(
                    label="📍 Download all structures",
                    data=zip_file,
                    file_name=f"{file_name}_curated.zip",
                    mime="application/zip",
                    key="download_all_structures",
                    on_click=lambda: st.session_state.update({"all_structures_downloaded": True}),
                )
                    