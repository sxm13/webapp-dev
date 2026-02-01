from streamlit_extras.mention import mention
from PIL import Image
import streamlit as st
from streamlit_extras.colored_header import colored_header

st.set_page_config(page_title="Workflow", page_icon="./figures/logo.png")
logo_url = "./figures/logo.png"
st.sidebar.image(logo_url)
st.logo(logo_url, link="https://zenodo.org/communities/core-mofs/records?q=&l=list&p=1&s=10&sort=newest")

st.title("Workflow")

with st.sidebar:
    st.markdown("© 2024 CoRE MOF Project. CC-BY-4.0 License.")


colored_header(
                    label="Collection",
                    description="",
                    color_name="orange-50",
                )

st.image("./figures/workflow1.png")

st.markdown('''
            -   Scrap digital object identifiers (DOIs) from [Web of Science (WOS)](https://www.webofscience.com/wos/woscc/basic-search) list
                using the keywords "metal-organic framework" or "porous organic polymers".
            -   Extract experimental CIFs from papers.
            -   Download CIFs from [Supplementary Information(SI)](https://zenodo.org/record/14510695) and [Cambridge Structural Database (CSD)](https://www.ccdc.cam.ac.uk/support-and-resources/downloads/).
            ''')
# st.markdown('''
#             -   Scrap digital object identifiers (DOIs) from [🤖 PapersBot](https://github.com/fxcoudert/PapersBot) list
#                 using the keywords "metal-organic framework" or "porous coordination polymers".
#             -   Extract experimental CIFs from papers.
#             -   Download CIFs from [Cambridge Structural Database (CSD)](https://www.ccdc.cam.ac.uk/solutions/software/csd/) 
#                 and Supplementary Information(SI).
#             ''')

colored_header(
                    label="Curation",
                    description="",
                    color_name="orange-50",
                )

st.image("./figures/workflow2.png")

mention(
    label="Script",
    icon="github",
    url="https://github.com/mtap-research/CoRE-MOF-Tools/blob/main/1-clean"
)
st.markdown('''
                -   Pre-check & Pre-process
                    -   Remove structures without metals or carbon
                    -   Split multi-structures from a single CIF.
                    -   Convert to a primitive cell.
                    -   Set symmetry to P1.
                ''')

split = "./figures/split.png"
image_split = Image.open(split)
original_width, original_height = image_split.size
scale_factor = 0.2
new_width = int(original_width * scale_factor)
st.image(split, width=new_width)

st.markdown('''
                -   Clean
                    -   FSR: Free Solvent Removed
                        -   Initial skin: 0.25 Å.
                        -   [CSD vdW radii](https://doi.org/10.1039/B801115J).
                    -   Keep Ions
                        -   [Ion list](https://github.com/mtap-research/CoRE-MOF-Tools/blob/main/1-clean/utils/ions_list.py)
                    -   ASR: All Solvent Removed
                ''')

clean = "./figures/clean.png"
image_clean = Image.open(clean)
original_width, original_height = image_clean.size
scale_factor = 0.2
new_width = int(original_width * scale_factor)
st.image(clean, width=new_width)


colored_header(
                    label="Checker",
                    description="",
                    color_name="orange-50",
                )

st.image("./figures/workflow3.png")

mention(
        label="Script",
        icon="github",
        url="https://github.com/mtap-research/CoRE-MOF-Tools/blob/main/2-NCRCheck/check.py"
    )
st.markdown('''
            -   Chen and Manz
                -   [Atom typing radii (ATR)](https://doi.org/10.1039/C9RA07327B)
                -   :page_with_curl: checking list :warning: : 
                    -   *isolated*
                    -   *overlapping atom*
                    -   *under bonded carbon*
                    -   *over bonded carbon*
            -   [mofchecker](https://github.com/kjappelbaum/mofchecker)
                -   :page_with_curl: checking list :warning: :
                    -   *has_atomic_overlaps*
                    -   *has_overcoordinated_c*  
                    -   *has_overcoordinated_n*
                    -   *has_overcoordinated_h*
                    -   *has_suspicious_terminal_oxo*
                    -   *has_undercoordinated_c*
                    -   *has_undercoordinated_n*
                    -   *has_undercoordinated_rare_earth*
                    -   *has_undercoordinated_alkali_alkaline*
                    -   *has_geometrically_exposed_metal*
                    -   *has_carbon*
                    -   *has_lone_molecule* (not used for structure with ions)
            ''')