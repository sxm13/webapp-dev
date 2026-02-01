import pandas as pd
import streamlit as st
# from streamlit_extras.markdownlit  import mdlit
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
                    page_title="CoRE MOF Database",
                    page_icon="🏘️",
                    layout="centered"
                    )

width = "100px"
height = "100px"
st.title("Welcome to CoRE MOF Database")

st.divider()

logo_url = "./figures/logo.png"
st.logo(logo_url, link="https://zenodo.org/communities/core-mofs/records?q=&l=list&p=1&s=10&sort=newest")
st.sidebar.image(logo_url)
st.sidebar.write("© 2025 CoRE MOF Project. CC-BY-4.0 License.")

st.markdown(
            """
            **Co**mputation-**R**eady **E**xperimental **M**etal-**O**rganic **F**ramework Database (CoRE MOF DB) includes more than 40,000 MOFs 
            reported up to December 31st, 2023 (manuscript acceptance date). CoRE MOF DB includes pre-computed geometric 
            and machine-learned properties, such as porosity, surface areas, DDEC06 partial atomic charges, thermal & 
            solvent removal stability.

            More details can be found in 
            *   Paper 👉 [Matter, 2025, 0 (0): 102140](https://doi.org/10.1016/j.matt.2025.102140).
            *   SI dataset 👉 [Zenodo](https://zenodo.org/records/15055758)
            *   CSD Modified dataset 👉 [Cambridge Structural Database (CSD)](https://www.ccdc.cam.ac.uk/support-and-resources/downloads/).
            *   CSD Unmodified dataset 👉 [Github script](https://github.com/ccdc-opensource/csd-python-api-scripts/tree/main/notebooks/CoRE-MOF)
            """
            )

add_vertical_space(2)

colored_header(
                label="Database Content",
                description="",
                color_name="orange-50",
            )

st.code('''   
        CoRE MOF 2024 Database      # Full Database (N = 40,837)
        │
        ├──SI Dataset               # source from Supporting Information (N = 8,300)
        │   │   
        │   ├── CR                  # computation-ready (N = 2,664)
        │   │   │
        │   │   ├── ASR             # all solvent removed (N = 1,372)
        │   │   ├── FSR             # free solvent removed (N = 1,192)
        │   │   └── Ion             # with ion (N = 100)
        │   │ 
        │   └── NCR                 # not computation-ready (N = 5,636)
        │
        ├── CSD Modified Dataset    # modified CIFs source from the CSD (N = 20,276)
        │   │
        │   ├── CR                  # (N = 9,835)
        │   │   │
        │   │   ├── ASR             # (N = 5,591)
        │   │   ├── FSR             # (N = 3,786)
        │   │   └── Ion             # (N = 458)
        │   │
        │   └── NCR                 # (N = 10,441)
        │
        └── CSD Unmodified Dataset  # unmodified CIFs source from the CSD (N = 12,261)
            │
            ├── CR                  # (N = 4,703)
            │   │
            │   ├── ASR             # (N = 1,894)
            │   ├── FSR             # (N = 2,657)
            │   └── Ion             # (N = 152)
            │
            └── NCR                 # (N = 7,558)
        ''', language='plaintext')


# colored_header(
#                 label="Data Source (Computation-Ready Set)",
#                 description="Total CSD / SI CIFs: 14,285 / 2,636",
#                 color_name="orange-50",
#             )

colored_header(
                label="Data Source ",
                description="The number of structures is obtained as the sum of the ASR and Ion datasets since there is a significant overlap in the ASR and FSR datasets (Total NCR / CR CIFs: 12,321 / 8,585).",
                color_name="orange-50",
            )

data_ = pd.read_csv("./data/internal/All_data_20241205.csv",low_memory=False)
# data_csd = all_data[all_data["Source"]=="CSD"]
# data_si = all_data[all_data["Source"]=="SI"]
# data_csd['Year'] = pd.to_numeric(data_csd['Year'], errors='coerce')
# data_si['Year'] = pd.to_numeric(data_si['Year'], errors='coerce') 
# data_csd = data_csd.dropna(subset=['Year'])
# data_si = data_si.dropna(subset=['Year'])
# data_csd['Year'] = data_csd['Year'].astype(int)
# data_si['Year'] = data_si['Year'].astype(int)
# xx = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 
#      2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
#      2020, 2021, 2022, 2023, 2024]
# y_c = []
# for i in range(len(xx)):
#     if xx[i] == 1995:
#         csd_count = data_csd[data_csd['Year'] <= 1995].shape[0]
#         si_count = data_si[data_si['Year'] <= 1995].shape[0]
#     else:
#         csd_count = data_csd[data_csd['Year'] == xx[i]].shape[0]
#         si_count = data_si[data_si['Year'] == xx[i]].shape[0]
#     y_c.append([xx[i], "CSD", csd_count])
#     y_c.append([xx[i], "SI", si_count])

# y_c = [
#     [1995, 'CSD', 15],
#     [1995, 'SI', 0],
#     [1996, 'CSD', 9],
#     [1996, 'SI', 0],
#     [1997, 'CSD', 8],
#     [1997, 'SI', 0],
#     [1998, 'CSD', 17],
#     [1998, 'SI', 0],
#     [1999, 'CSD', 28],
#     [1999, 'SI', 0],
#     [2000, 'CSD', 35],
#     [2000, 'SI', 0],
#     [2001, 'CSD', 54],
#     [2001, 'SI', 1],
#     [2002, 'CSD', 79],
#     [2002, 'SI', 0],
#     [2003, 'CSD', 132],
#     [2003, 'SI', 1],
#     [2004, 'CSD', 143],
#     [2004, 'SI', 2],
#     [2005, 'CSD', 247],
#     [2005, 'SI', 0],
#     [2006, 'CSD', 309],
#     [2006, 'SI', 3],
#     [2007, 'CSD', 399],
#     [2007, 'SI', 25],
#     [2008, 'CSD', 502],
#     [2008, 'SI', 8],
#     [2009, 'CSD', 438],
#     [2009, 'SI', 5],
#     [2010, 'CSD', 642],
#     [2010, 'SI', 8],
#     [2011, 'CSD', 850],
#     [2011, 'SI', 9],
#     [2012, 'CSD', 967],
#     [2012, 'SI', 22],
#     [2013, 'CSD', 987],
#     [2013, 'SI', 40],
#     [2014, 'CSD', 1060],
#     [2014, 'SI', 70],
#     [2015, 'CSD', 1174],
#     [2015, 'SI', 140],
#     [2016, 'CSD', 1145],
#     [2016, 'SI', 171],
#     [2017, 'CSD', 1134],
#     [2017, 'SI', 352],
#     [2018, 'CSD', 729],
#     [2018, 'SI', 188],
#     [2019, 'CSD', 802],
#     [2019, 'SI', 434],
#     [2020, 'CSD', 779],
#     [2020, 'SI', 437],
#     [2021, 'CSD', 474],
#     [2021, 'SI', 252],
#     [2022, 'CSD', 406],
#     [2022, 'SI', 130],
#     [2023, 'CSD', 601],
#     [2023, 'SI', 171],
#     [2024, 'CSD', 120],
#     [2024, 'SI', 167]
#     ]

y_c = [
    [1995, 'NCR', 90],
    [1995, 'CR', 6],
    [1996, 'NCR', 112],
    [1996, 'CR', 9],
    [1997, 'NCR', 145],
    [1997, 'CR', 12],
    [1998, 'NCR', 172],
    [1998, 'CR', 21],
    [1999, 'NCR', 194],
    [1999, 'CR', 35],
    [2000, 'NCR', 255],
    [2000, 'CR', 52],
    [2001, 'NCR', 322],
    [2001, 'CR', 81],
    [2002, 'NCR', 417],
    [2002, 'CR', 119],
    [2003, 'NCR', 484],
    [2003, 'CR', 185],
    [2004, 'NCR', 622],
    [2004, 'CR', 260],
    [2005, 'NCR', 733],
    [2005, 'CR', 387],
    [2006, 'NCR', 922],
    [2006, 'CR', 548],
    [2007, 'NCR', 1102],
    [2007, 'CR', 759],
    [2008, 'NCR', 1361],
    [2008, 'CR', 1020],
    [2009, 'NCR', 1654],
    [2009, 'CR', 1247],
    [2010, 'NCR', 2002],
    [2010, 'CR', 1587],
    [2011, 'NCR', 2441],
    [2011, 'CR', 2007],
    [2012, 'NCR', 2969],
    [2012, 'CR', 2532],
    [2013, 'NCR', 3501],
    [2013, 'CR', 3088],
    [2014, 'NCR', 4300],
    [2014, 'CR', 3674],
    [2015, 'NCR', 5174],
    [2015, 'CR', 4323],
    [2016, 'NCR', 6108],
    [2016, 'CR', 5010],
    [2017, 'NCR', 7336],
    [2017, 'CR', 5728],
    [2018, 'NCR', 8124],
    [2018, 'CR', 6194],
    [2019, 'NCR', 9204],
    [2019, 'CR', 6825],
    [2020, 'NCR', 10182],
    [2020, 'CR', 7438],
    [2021, 'NCR', 10817],
    [2021, 'CR', 7805],
    [2022, 'NCR', 11217],
    [2022, 'CR', 8082],
    [2023, 'NCR', 11954],
    [2023, 'CR', 8459],
    [2024, 'NCR', 12321],
    [2024, 'CR', 8585]
    ]


st.bar_chart(pd.DataFrame(y_c, columns=["Year", "CoRE MOF Database", "Cumulative CIFs"]),
                            x="Year", y="Cumulative CIFs", color="CoRE MOF Database")


def create_pie_chart_figure(
    df,
    extension_label: str,
    title: str = "CoRE MOF 2024"
):

    df_CSD = df[df["Source"] == "CSD"]
    df_SI = df[df["Source"] == "SI"]

    counts_CSD = df_CSD["Publication"].value_counts()
    counts_SI = df_SI["Publication"].value_counts()

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])

    fig.add_trace(
        go.Pie(
            labels=counts_CSD.index,
            values=counts_CSD.values,
            name="CSD"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Pie(
            labels=counts_SI.index,
            values=counts_SI.values,
            name="SI"
        ),
        row=1, col=2
    )

    fig.update_traces(hole=0.4, hoverinfo="label+percent+name")

    fig.update_layout(
        # title_text=f"{title} {extension_label}",
        annotations=[
            dict(
                text='CSD',
                x=0.18,
                y=0.5,
                font_size=20,
                showarrow=False
            ),
            dict(
                text='SI',
                x=0.79,
                y=0.5,
                font_size=20,
                showarrow=False
            )
        ]
    )
    return fig

fig_ASR = create_pie_chart_figure(data_, extension_label="")
st.plotly_chart(fig_ASR, use_container_width=True)

# df_ASR = data_[data_["Extension"]=="All Solvent Removed"]
# fig_ASR = create_pie_chart_figure(df_ASR, extension_label="ASR")
# st.plotly_chart(fig_ASR, use_container_width=True)

# df_FSR = data_[data_["Extension"]=="Free Solvent Removed"]
# fig_FSR = create_pie_chart_figure(df_FSR, extension_label="FSR")
# st.plotly_chart(fig_FSR, use_container_width=True)

# df_Ion = data_[data_["Extension"]=="with ion"]
# fig_Ion = create_pie_chart_figure(df_Ion, extension_label="Ion")
# st.plotly_chart(fig_Ion, use_container_width=True)



# colored_header(
#         label="Contributors",
#         description="",
#         color_name="orange-50",
#     )
# mdlit(
#         """[MTAP Group](https://sites.google.com/view/mtap-lab)
#         """
#         )
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     st.image("./figures/contributors/YongchulChung.png")
# with col2:
#     st.image("./figures/contributors/ZhaoGuobin.png")
# with col3:
#     st.image("./figures/contributors/haewonkim.jpg")
# with col4:
#     st.image("./figures/contributors/seonghyeon.jpg")
# with col5:
#     st.image("./figures/contributors/chenyu.jpg")
# mdlit(
#         """[Sholl Research Group](https://sholl.chbe.gatech.edu/index.html)
#         """
#         )
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     st.image("./figures/contributors/davidsholl.jpg")
# with col2:
#     st.image("./figures/contributors/logan.png")    
# mdlit(
#         """[The Siepmann Group](https://siepmann.chem.umn.edu/)
#         """
#         )
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     st.image("./figures/contributors/siepmann.jpg")
# with col2:
#     st.image("./figures/contributors/saumil_chheda.jpg")  
# with col3:
#     st.image("./figures/contributors/prerna.jpg")
# mdlit(
#         """[Moosavi's GROUP](https://chem-eng.utoronto.ca/faculty-staff/faculty-members/seyed-mohamad-moosavi/)
#         """
#         )
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     st.image("./figures/contributors/seyed-mohamad-moosavi.png")
# with col2:
#     st.image("./figures/contributors/juhuang.jpg")  
# mdlit(
#         """[Snurr Research Group](https://zeolites.cqe.northwestern.edu/)
#         """
#         )
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     st.image("./figures/contributors/Randy.jpg")
# with col2:
#     st.image("./figures/contributors/Kunhuan_Liu.jpg")  
# with col3:
#     st.image("./figures/contributors/kenji.jpg")  
# with col4:
#     st.image("./figures/contributors/thang_pham.jpg")  
# mdlit(
#         """[Kulik Research Group](https://hjkgrp.mit.edu/)
#         """
#         )
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     st.image("./figures/contributors/kulik.png")
# with col2:
#     st.image("./figures/contributors/gianmarco.jpg")  
# mdlit(
#         """[François-Xavier Coudert Research group](https://www.coudert.name/group.html)
#         """
#         )
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     st.image("./figures/contributors/FX.jpg")
# with col2:
#     st.image("./figures/contributors/Zoubritzky.jpg")  
# mdlit(
#         """[Dr. Haranczyk](https://materials.imdea.org/people/maciej-haranczyk/)
#         """
#         )
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     st.image("./figures/contributors/Haranczyk.jpg")       

colored_header(
                label="Visitor Map Tracker",
                description="",
                color_name="orange-50",
            )


st.markdown("""
            <a href='https://mapmyvisitors.com/web/1bxpv'  title='Visit tracker'><img src='https://mapmyvisitors.com/map.png?cl=f4f0f0&w=1200&t=tt&d=YzF3Izguaz05mAnMNXYhCW7GlWX8nWq9pqjl3r_0YzQ&co=bcccd8&ct=110707'/></a>
            """, unsafe_allow_html=True)
