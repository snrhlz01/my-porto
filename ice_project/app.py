import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from typing import Dict
from typing import Set
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# MATRIX FUNCTION
def format_big_number(num):
        if num >= 1e6:
                return f"{num / 1e6:.2f} Mio"
        elif num >= 1e3:
                return f"{num / 1e3:.2f} K"
        else:
                return f"{num:.0f}"

# FUNCTION
def format_big_number(num):
    if num >= 1e6:
        return f"{num / 1e6:.2f} Mio"
    elif num >= 1e3:
        return f"{num / 1e3:.2f} K"
    else:
        return f"{num:.0f}"

def get_first_largest(x):
    unique_values = sorted(set(x), reverse=True)
    return unique_values[0] if unique_values else None

def get_second_largest(x):
    unique_values = sorted(set(x), reverse=True)
    if len(unique_values) > 1:
        return unique_values[1]
    else:
        return None

def get_third_largest(x):
    unique_values = sorted(set(x), reverse=True)
    if len(unique_values) > 2:
        return unique_values[2]
    else:
        return None

def get_fourth_largest(x):
    unique_values = sorted(set(x), reverse=True)
    if len(unique_values) > 3:
        return unique_values[3]
    else:
        return None

def get_fifth_largest(x):
    unique_values = sorted(set(x), reverse=True)
    if len(unique_values) > 4:
        return unique_values[4]
    else:
        return None


# FUNCTION IN WORKLOAD DETAIL
def adjust_duplicate_ids(df):
    unique_ids = set()
    for i, row in df.iterrows():
        while row['id_'] in unique_ids:
            # Increment num_events for the current row
            df.at[i, 'num_events'] += 1
            # Update the id_ with the new num_events value
            row['id_'] = str(row['date']) + "_" + str(df.at[i, 'num_events'])
            df.at[i, 'id_'] = row['id_']
        unique_ids.add(row['id_'])
    return df

def initialize_state():
    """Initializes all filters and counter in Streamlit Session State
    """
    for q in ["category_act"]:
        if f"{q}_query" not in st.session_state:
            st.session_state[f"{q}_query"] = set()

def filter_top_categories(df, top_categories, month_year_filter):
    """
    Filter data frame based on top categories and specified month and year.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing event data.
        top_categories (pandas.DataFrame): DataFrame containing top categories.
        month_year_filter (str): String representing the month and year to filter (format: 'Month Year').

    Returns:
        pandas.DataFrame: Filtered DataFrame containing data for top categories in the specified month and year.
    """

    # Count number of events per week per category
    num_events_per_week = df.groupby(['date', 'category_act'])['act_id'].count().reset_index(name='num_events')

    # Merge with top_categories to get the top categories
    top_categories_week = pd.merge(top_categories, num_events_per_week, on='category_act', how='left')

    # Rename columns and drop duplicates
    top_categories_week = top_categories_week.drop(columns=['num_events_x', 'bulan_tahun', 'urut_bulan_tahun'])
    top_categories_week = top_categories_week.rename(columns={"num_events_y": "num_events"})
    top_categories_week = top_categories_week.drop_duplicates(subset=['category_act', 'date'])
    top_categories_week['date'] = pd.to_datetime(top_categories_week['date']).dt.date
    top_categories_week['bulan_tahun'] = top_categories_week['date'].apply(lambda x: x.strftime('%B %Y'))

    # Filter data for the selected month and year
    filtered_data = top_categories_week[top_categories_week['bulan_tahun'] == month_year_filter]

    # Convert date to datetime
    filtered_data['date'] = pd.to_datetime(filtered_data['date'])

    # Group by 'category_act' and 'date' with a weekly frequency
    weekly_data = filtered_data.groupby(['category_act', pd.Grouper(key='date', freq='W')])['num_events'].sum().reset_index()

    # Take top 5 categories
    top5_categories = weekly_data.groupby('category_act')['num_events'].sum().nlargest(5).index

    # Filter data to include only top 5 categories
    filtered_data_top5 = weekly_data[weekly_data['category_act'].isin(top5_categories)]
    
    filtered_data_top5['id_'] = filtered_data_top5['date'].astype(str) + "_" + filtered_data_top5['num_events'].astype(str)
    filtered_data_top5 = adjust_duplicate_ids(filtered_data_top5)

    return filtered_data_top5


def query_data(df):
    """Apply filters in Streamlit Session State
    to filter the input DataFrame
    """
    df["category_act_id"] = df["date"].astype(str) + "_" + df["num_events"].astype(str)
    df["selected"] = True

    for q in ["category_act_id"]:
        if st.session_state[f"{q}_query"]:
            df.loc[~df[q].isin(st.session_state[f"{q}_query"]), "selected"] = False

    return df

def build_line_chart(df) -> go.Figure:
    
    fig = px.line(df, x="date", y="num_events", color='category_act', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_xaxes(
        title_text='',
        tickvals=pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='W-MON'),
        tickformat="%d %b"
    )

    # Atur tampilan sumbu y
    fig.update_yaxes(title_text='')
    fig.update_traces(mode='lines+markers+text')
    # Atur layout
    fig.update_layout(
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=1.2, 
            xanchor="left", 
            x=0, 
            font=dict(size=12)
        ),
        margin=dict(t=100),
        clickmode='event+select',
       
        autosize=True
    )

    for trace in fig.data:
        trace.text = trace.y  # Atur teks pada setiap titik sama dengan nilai y
        trace.hoverinfo = 'text'  # Hanya tampilkan teks saat dihover
        trace.textposition = 'top center'  # Atur posisi teks di atas setiap titik
    
    return fig

def render_plotly_ui(df):
    
    container0, container1, container2, container3, container4, container5 = st.columns(6)

    with container0:
            st.subheader(f"Top 5 Activity in :blue[{month_year_filter}]")
        #     year_filter = st.selectbox("Select Year", df['year'].unique())
            filtered_df = df[df['year'] == year_filter]

        #     month_year_filter = st.selectbox("Select Month and Year", filtered_df['bulan_tahun'].unique())
            filtered_df = filtered_df[filtered_df['bulan_tahun'] == month_year_filter]
            CURR_MONTH = filtered_df['urut_bulan_tahun'].max()
            PREV_MONTH = CURR_MONTH - 1

            num_events_per_date = df.groupby(['bulan_tahun', 'urut_bulan_tahun', 'category_act'])['act_id'].count().reset_index(name='num_events')
            top_categories = num_events_per_date.groupby(['bulan_tahun', 'urut_bulan_tahun']).apply(lambda x: x.nlargest(10, 'num_events')).reset_index(drop=True)
            num_events_per_date_ = df.groupby(['bulan_tahun', 'urut_bulan_tahun', 'category_act'])['act_id'].count().reset_index(name='num_events')
            top_categories_ = pd.merge(top_categories, num_events_per_date_, on='category_act', how='left')
            top_categories_ = top_categories_.drop_duplicates(subset=['category_act', 'bulan_tahun_x'])
            top_categories_ = top_categories_.dropna()
            top_categories_ = top_categories_.drop(columns=['bulan_tahun_y', 'urut_bulan_tahun_y', 'num_events_x'])
            top_categories_ = top_categories_.rename(columns={"bulan_tahun_x": "bulan_tahun",
                                                                    "urut_bulan_tahun_x": "urut_bulan_tahun",
                                                                    "num_events_y": "num_events"})

            top_categories_ = top_categories_[top_categories_['bulan_tahun'] == month_year_filter]
        
    with container1:
        data = pd.pivot_table(
            data=df,
            index=['urut_bulan_tahun', 'category_act'],
            aggfunc={'act_id': pd.Series.nunique}
        ).reset_index()
        data_ = pd.merge(data, top_categories, on='urut_bulan_tahun')
        data_ = data_.drop(columns=['category_act_x', 'act_id'])
        data_ = data_.rename(columns={'category_act_y': "category_act"})
        data_ = data_.drop_duplicates(subset=['urut_bulan_tahun', 'category_act'])
        max_category_act = data_.groupby('urut_bulan_tahun')['num_events'].agg(get_first_largest).reset_index()
        first_max_category_data = pd.merge(data_, max_category_act, on='urut_bulan_tahun', suffixes=('', '_second'))
        first_max_category_data = first_max_category_data[first_max_category_data['bulan_tahun'] == month_year_filter]
        first_max_category_data = first_max_category_data.loc[first_max_category_data['num_events'] == first_max_category_data['num_events_second']]
        first_max_category_data = first_max_category_data['category_act'].iloc[0]
        data_ = data_[data_['category_act'] == first_max_category_data]
        curr_sales = data_.loc[data_['urut_bulan_tahun'] == CURR_MONTH, 'num_events'].values[0] if CURR_MONTH in data_['urut_bulan_tahun'].values else 0
        prev_sales = data_.loc[data_['urut_bulan_tahun'] == PREV_MONTH, 'num_events'].values[0] if PREV_MONTH in data_['urut_bulan_tahun'].values else 0

        sales_diff_pct = 100.0 * (curr_sales - prev_sales) / prev_sales if prev_sales != 0 else 0
        st.metric(f"Top 1: **{first_max_category_data.title()}**", value=format_big_number(curr_sales), delta=f"{sales_diff_pct:.2f}% vs Last Month")

    with container2:
        data = pd.pivot_table(
            data=df,
            index=['urut_bulan_tahun', 'category_act'],
            aggfunc={'act_id': pd.Series.nunique}
        ).reset_index()
        data_ = pd.merge(data, top_categories, on='urut_bulan_tahun')
        data_ = data_.drop(columns=['category_act_x', 'act_id'])
        data_ = data_.rename(columns={'category_act_y': "category_act"})
        data_ = data_.drop_duplicates(subset=['urut_bulan_tahun', 'category_act'])
        second_max_values = data_.groupby('urut_bulan_tahun')['num_events'].agg(get_second_largest).reset_index()
        second_max_category_data = pd.merge(data_, second_max_values, on='urut_bulan_tahun', suffixes=('', '_second'))
        second_max_category_data = second_max_category_data[second_max_category_data['bulan_tahun'] == month_year_filter]
        second_max_category_data = second_max_category_data.loc[second_max_category_data['num_events'] == second_max_category_data['num_events_second']]
        second_max_category_data = second_max_category_data['category_act'].iloc[0]
        data_ = data_[data_['category_act'] == second_max_category_data]
        curr_sales = data_.loc[data_['urut_bulan_tahun'] == CURR_MONTH, 'num_events'].values[0] if CURR_MONTH in data_['urut_bulan_tahun'].values else 0
        prev_sales = data_.loc[data_['urut_bulan_tahun'] == PREV_MONTH, 'num_events'].values[0] if PREV_MONTH in data_['urut_bulan_tahun'].values else 0

        sales_diff_pct = 100.0 * (curr_sales - prev_sales) / prev_sales if prev_sales != 0 else 0
        st.metric(f"Top 2: **{second_max_category_data.title()}**", value=format_big_number(curr_sales), delta=f"{sales_diff_pct:.2f}% vs Last Month")

    with container3:
        data = pd.pivot_table(
            data=df,
            index=['urut_bulan_tahun', 'category_act'],
            aggfunc={'act_id': pd.Series.nunique}
        ).reset_index()
        data_ = pd.merge(data, top_categories, on='urut_bulan_tahun')
        data_ = data_.drop(columns=['category_act_x', 'act_id'])
        data_ = data_.rename(columns={'category_act_y': "category_act"})
        data_ = data_.drop_duplicates(subset=['urut_bulan_tahun', 'category_act'])
        third_max_values = data_.groupby('urut_bulan_tahun')['num_events'].agg(get_third_largest).reset_index()
        third_max_category_data = pd.merge(data_, third_max_values, on='urut_bulan_tahun', suffixes=('', '_third'))
        third_max_category_data = third_max_category_data[third_max_category_data['bulan_tahun'] == month_year_filter]
        third_max_category_data = third_max_category_data.loc[third_max_category_data['num_events'] == third_max_category_data['num_events_third']]
        third_max_category_data = third_max_category_data['category_act'].iloc[0]
        data_ = data_[data_['category_act'] == third_max_category_data]
        curr_sales = data_.loc[data_['urut_bulan_tahun'] == CURR_MONTH, 'num_events'].values[0] if CURR_MONTH in data_['urut_bulan_tahun'].values else 0
        prev_sales = data_.loc[data_['urut_bulan_tahun'] == PREV_MONTH, 'num_events'].values[0] if PREV_MONTH in data_['urut_bulan_tahun'].values else 0

        sales_diff_pct = 100.0 * (curr_sales - prev_sales) / prev_sales if prev_sales != 0 else 0
        st.metric(f"Top 3: **{third_max_category_data.title()}**", value=format_big_number(curr_sales), delta=f"{sales_diff_pct:.2f}% vs Last Month")

    with container4:
        data = pd.pivot_table(
            data=df,
            index=['urut_bulan_tahun', 'category_act'],
            aggfunc={'act_id': pd.Series.nunique}
        ).reset_index()
        data_ = pd.merge(data, top_categories, on='urut_bulan_tahun')
        data_ = data_.drop(columns=['category_act_x', 'act_id'])
        data_ = data_.rename(columns={'category_act_y': "category_act"})
        data_ = data_.drop_duplicates(subset=['urut_bulan_tahun', 'category_act'])
        fourth_max_values = data_.groupby('urut_bulan_tahun')['num_events'].agg(get_fourth_largest).reset_index()
        fourth_max_category_data = pd.merge(data_, fourth_max_values, on='urut_bulan_tahun', suffixes=('', '_fourth'))
        fourth_max_category_data = fourth_max_category_data[fourth_max_category_data['bulan_tahun'] == month_year_filter]
        fourth_max_category_data = fourth_max_category_data.loc[fourth_max_category_data['num_events'] == fourth_max_category_data['num_events_fourth']]
        fourth_max_category_data = fourth_max_category_data['category_act'].iloc[0]
        data_ = data_[data_['category_act'] == fourth_max_category_data]
        curr_sales = data_.loc[data_['urut_bulan_tahun'] == CURR_MONTH, 'num_events'].values[0] if CURR_MONTH in data_['urut_bulan_tahun'].values else 0
        prev_sales = data_.loc[data_['urut_bulan_tahun'] == PREV_MONTH, 'num_events'].values[0] if PREV_MONTH in data_['urut_bulan_tahun'].values else 0

        sales_diff_pct = 100.0 * (curr_sales - prev_sales) / prev_sales if prev_sales != 0 else 0
        st.metric(f"Top 4: **{fourth_max_category_data.title()}**", value=format_big_number(curr_sales), delta=f"{sales_diff_pct:.2f}% vs Last Month")

    with container5:
        data = pd.pivot_table(
            data=df,
            index=['urut_bulan_tahun', 'category_act'],
            aggfunc={'act_id': pd.Series.nunique}
        ).reset_index()
        data_ = pd.merge(data, top_categories, on='urut_bulan_tahun')
        data_ = data_.drop(columns=['category_act_x', 'act_id'])
        data_ = data_.rename(columns={'category_act_y': "category_act"})
        data_ = data_.drop_duplicates(subset=['urut_bulan_tahun', 'category_act'])
        fifth_max_values = data_.groupby('urut_bulan_tahun')['num_events'].agg(get_fifth_largest).reset_index()
        fifth_max_category_data = pd.merge(data_, fifth_max_values, on='urut_bulan_tahun', suffixes=('', '_fifth'))
        fifth_max_category_data = fifth_max_category_data[fifth_max_category_data['bulan_tahun'] == month_year_filter]
        fifth_max_category_data = fifth_max_category_data.loc[fifth_max_category_data['num_events'] == fifth_max_category_data['num_events_fifth']]
        fifth_max_category_data = fifth_max_category_data['category_act'].iloc[0]
        data_ = data_[data_['category_act'] == fifth_max_category_data]
        curr_sales = data_.loc[data_['urut_bulan_tahun'] == CURR_MONTH, 'num_events'].values[0] if CURR_MONTH in data_['urut_bulan_tahun'].values else 0
        prev_sales = data_.loc[data_['urut_bulan_tahun'] == PREV_MONTH, 'num_events'].values[0] if PREV_MONTH in data_['urut_bulan_tahun'].values else 0

        sales_diff_pct = 100.0 * (curr_sales - prev_sales) / prev_sales if prev_sales != 0 else 0
        st.metric(f"Top 5: **{fifth_max_category_data.title()}**", value=format_big_number(curr_sales), delta=f"{sales_diff_pct:.2f}% vs Last Month")
    
    df = filter_top_categories(df, top_categories, month_year_filter)
    line_chart = build_line_chart(df)
    selected_points = plotly_events(
        line_chart,
        click_event=True,
        select_event=False,  # Ensure only click events are considered
    hover_event=False
    )
     
    current_query = {}
    current_query['category_act_query'] = {
            f"{el['x']}_{el['y']}"
            for el in selected_points
        }
   
    return {'df': df, 'current_query': current_query, 'top_categories_': top_categories_,"month_year_filter":month_year_filter}


def update_state(current_query: Dict[str, Set]):
    """Stores input dict of filters into Streamlit Session State.

    If one of the input filters is different from previous value in Session State, 
    rerun Streamlit to activate the filtering and plot updating with the new info in State.
    """
    rerun = False
    for q in ["category_act"]:
        if current_query[f"{q}_query"] - st.session_state[f"{q}_query"]:
            st.session_state[f"{q}_query"] = current_query[f"{q}_query"]
            rerun = True

    if rerun:
        st.experimental_rerun()

    # Simpan data yang dipilih oleh pengguna
    selected_data = []
    for el in current_query['category_act_query']:
        date_ = el
        selected_data.append((date_))
    st.session_state.selected_data = selected_data

def display_selected_data():
  if 'selected_data' in st.session_state:
    selected_data = st.session_state.selected_data
    return selected_data
  else:
    return None



# WEBPAGE CONFIG
st.set_page_config(layout='wide',
                page_title='Workload Analysis in X Company)',
                page_icon=':office',
                )

st.markdown(
    """
    <style>
    .main {
        max-width: 100%;
        margin: 0 auto;
        padding: 1rem;
    }
    @media (max-width: 768px) {
        .main {
            font-size: 14px;
            padding: 0.5rem;
        }
    }
    @media (max-width: 480px) {
        .main {
            font-size: 12px;
            padding: 0.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# st.markdown(
#     """
#     <div style='text-align: center;'>
#         <img src='https://github.com/snrhlz01/workload_analysis/raw/main/logo.png' class='responsive-img' width='150'>
#     </div>
#     """,
#     unsafe_allow_html=True
# )


# HEADER SECTION
st.markdown("""
                <div style='text-align:center'>
                <h2>Workload Analysis in X Company</h2>
                </div>
                """, unsafe_allow_html=True)

# SIDE SECTION
# with st.sidebar:
#         

# INITIALIZE SESSION STATE FOR NAVIGATION IF NOT SET
# if 'page' not in st.session_state:
#     st.session_state.page = 'Main'
page = "Main"

# # SIDEBAR FOR PAGE NAVIGATION
# analysis = st.sidebar.selectbox("Dashboard", ['Main', 'Workload Detail', 'Manpower Detail'], index=['Main', 'Workload Detail', 'Manpower Detail'].index(st.session_state.page))

# # UPDATE SESSION STATE BASED ON SIDEBAR SELECTION
# st.session_state.page = analysis


# LOAD DATASET
df = pd.read_excel('data/gitfile2 (1).xlsx')

# Key of Selectbox 
key1 = "selectbox_dashboard_year"
key2 = "selectbox_dashboard_month"
key3 = "selectbox_workload_year"
key4 = "selectbox_workload_month"
# act_id DROP DUPLICATE
# df = df.drop_duplicates(subset='act_id')

# remarks COLUMN FILL NULL VALUE WITH "done" or "not done"
df['remarks'] = df['remarks'].fillna('done')
# df['remarks'] = df['remarks'].fillna('not done')

# MAIN SECTION
if page == 'Main':
        # CREATE TAB
        # pendahuluan, dashboard,workload_detail, manpower = st.tabs(['Pendahuluan', 'Dashboard','Workload Detail' ,'Manpower'])
        dashboard,workload_detail, manpower = st.tabs(['Dashboard','Workload Detail' ,'Manpower'])

        ## PENDAHULUAN ##
        # with pendahuluan:
        #         pendahuluan_image, pendahuluan_write = st.columns(2)
        #         with pendahuluan_image:
        #                 img=Image.open(r"images/indonesia-convention.jpg")
        #                 st.image(img)

        #         with pendahuluan_write:
        #                 st.write("""
        #                         Indonesia Convention Exhibition (ICE atau ICE BSD City) merupakan pusat konvensi dan pameran terbesar di Indonesia. ICE yang terletak di komplek Bumi Serpong Damai (BSD), Kec. Pagedangan, Kab. Tangerang, Banten juga dikenal sebagai gedung konser dan gedung pernikahan terbesar. Gedung yang dibangun atas kerja sama perusahaan Sinar Mas Land dan Kompas Gramedia ini dioperasikan oleh perusahaan yang Bernama PT Indonesia International Expo.
        #                         """)
        #                 st.write("""
        #                         ICE merupakan salah satu gedung pameran terbesar di Asia Tenggara mempunyai luas lahan sebesar 22 Ha (220.000 m²) yang terdiri dari 10 ruang pameran dengan masing-masing ruang seluas sebesar 50.000 m² dan mempunyai tambahan ruang pameran luar ruangan seluas 50.000 m², aula konvensi seluas 4.000 m² yang dapat dibagi menjadi 4 ruangan, 33 ruang pertemuan,dan lobi pra-fungsi yang nyaman seluas 12.000 m².
        #                         """)
        #                 st.write("""
        #                         ICE dapat dipakai sebagai tempat penyelenggaraan berbagai acara berskala nasional dan internasional seperti pameran, konferensi, seminar, pertemuan, acara olahraga, pernikahan, konser, pesta rakyat, dan sebagainya. Salah satu acara terbesar yaitu GIIAS (Gaikindo Indonesia International Auto Show) yang merupakan pameran otomotif terbesar di Indonesia yang digelar setiap tahunnya di ICE sejak 2015.
        #                         """)

        ## DASHBOARD ##
        with dashboard:
                # 1st ROW 
                container1, container2_, container3_, container4_ = st.columns((0.45,0.6,1,0.6))
                
                container_height = "5px"  # Misalnya, set ke 500px atau nilai yang sesuai

                # Gunakan CSS styling untuk mengatur tinggi container2_, container3_, dan container4_
                container2_.markdown(f'<style>div.row-widget.stHorizontal > div:nth-of-type(2) {{height: {container_height} !important}}</style>', unsafe_allow_html=True)
                container3_.markdown(f'<style>div.row-widget.stHorizontal > div:nth-of-type(3) {{height: {container_height} !important}}</style>', unsafe_allow_html=True)
                container4_.markdown(f'<style>div.row-widget.stHorizontal > div:nth-of-type(4) {{height: {container_height} !important}}</style>', unsafe_allow_html=True)

                
                with container1:
                        # if st.button("Reset"):
                        # # Reset the selected year and month
                        #         year_filter = '2023'
                        #         month_year_filter = 'January 2023'
                        #         st.experimental_rerun()
                        year_filter = st.selectbox("Select Year", df['year'].unique())
                        filtered_df = df[df['year'] == year_filter]

                        # Filter dan urutkan DataFrame berdasarkan bulan dan tahun yang dipilih
                        month_year_filter = st.selectbox("Select Month and Year", filtered_df['bulan_tahun'].unique())
                        filtered_df = filtered_df[filtered_df['bulan_tahun'] == month_year_filter]

                        # Mendapatkan urutan bulan dan tahun saat ini dan bulan sebelumnya
                        CURR_MONTH = filtered_df['urut_bulan_tahun'].max()
                        PREV_MONTH = CURR_MONTH - 1

                        # Membuat tabel pivot untuk menghitung jumlah act_id unik per bulan
                        data = pd.pivot_table(
                        data=df,
                        index='urut_bulan_tahun',
                        aggfunc={'act_id': pd.Series.nunique}
                        ).reset_index()

                        # Mendapatkan jumlah act_id unik untuk bulan ini dan bulan sebelumnya
                        curr_sales = data.loc[data['urut_bulan_tahun'] == CURR_MONTH, 'act_id'].values[0] if CURR_MONTH in data['urut_bulan_tahun'].values else 0 
                        prev_sales = data.loc[data['urut_bulan_tahun'] == PREV_MONTH, 'act_id'].values[0] if PREV_MONTH in data['urut_bulan_tahun'].values else 0

                        sales_diff_pct = 100.0 * (curr_sales - prev_sales) / prev_sales

                        # Menampilkan metrik workload
                        st.metric("Total Workload", value=format_big_number(curr_sales), delta=f"{sales_diff_pct:.2f}% vs Last Month")         

                with container2_:
                        container1, container2 = st.columns([0.7,0.4])
                        df_e = pd.read_excel('data/gitfile2 (1).xlsx')
                        df_e = df_e.drop_duplicates(subset=['event_name','month'])
                        # df_e['event_name'] = df_e['event_name'].dropna()
                        df_ = df_e[df_e['year'] == year_filter]
                        df1_ = df_[['event_name','week_label','week']]
                        filtered_df = df_[df_['bulan_tahun'] == month_year_filter]
                        num_events_per_date = filtered_df.groupby(['week'])['event_name'].count().reset_index(name='num_events')
                        num_events_per_date = pd.merge(num_events_per_date, df1_, on=['week'], how='left')
                        num_events_per_date = num_events_per_date.drop(columns=["event_name"])
                        # num_events_per_date = num_events_per_date.rename(columns={'event_name_x': "event_name"})
                        num_events_per_date = num_events_per_date.drop_duplicates(subset=['week_label'])
                        num_events_per_date = num_events_per_date.drop_duplicates(subset=['week'])
                        total_events = num_events_per_date['num_events'].sum()
                        
                        with container1:
                                st.subheader("Trend of Events")
                                # st.write("")
                        with container2:
                                st.metric("Total Events", total_events)
                                # st.write("")
                                # st.write(num_events_per_date)
                        
                        # Plot tren dengan warna kuning dan fill di bawah garis
                        fig = px.line(num_events_per_date, x="week_label", y="num_events", color_discrete_sequence=['#377dff'])
                        fig.update_traces(fill='tonexty')  # Mengisi di bawah garis
                        fig.update_xaxes(title_text='', tickfont=dict(size=15))
                        fig.update_yaxes(title_text='')
                        fig.update_layout(height=200)
                        st.plotly_chart(fig, use_container_width=True)                        
                
                with container3_:
                        container1, container2 = st.columns([0.5,0.1])
                        with container1:
                                st.subheader("Trend of Status Time")
                                st.write("")
                                st.write("")
                        # Filter the dataframe to only include the events from the selected month and year
                        df_ = df.drop_duplicates(subset=["event_id"])
                        filtered_df = df_[df_['bulan_tahun'] == month_year_filter]

                        # Group the data by date and status date, and count the number of events for each group
                        num_events_per_date = filtered_df.groupby(['date','status_date'])['event_id'].count().reset_index(name='num_events')
                        
                        fig = px.line(num_events_per_date, x="date", y="num_events", color='status_date')
                        # Fill below the line
                        fig.update_xaxes(title_text='', tickfont=dict(size=15))
                        fig.update_yaxes(title_text='')
                        fig.update_layout(
                                legend=dict(
                                        orientation="h",
                                        yanchor="top",
                                        y=2,
                                        xanchor="center",
                                        x=0.5,
                                        font=dict(size=13)  # Ubah ukuran font sesuai dengan preferensi Anda
                                ),
                                margin=dict(t=50)
)
                        fig.update_layout(height=200)
                        # Display the chart using Streamlit
                        st.plotly_chart(fig, use_container_width=True)
                        
                with container4_:
                        container1, container2 = st.columns([0.9,0.5])
                        df_ = df.drop_duplicates(subset=['date',"employee_name"])
                        filtered_df = df_[df_['bulan_tahun'] == month_year_filter]
                        num_events_per_date = filtered_df.groupby('date')['employee_name'].count().reset_index(name='num_workers')
                        total_events = num_events_per_date['num_workers'].sum()
                        with container1:
                                st.subheader("Trend of Workers")
                                st.write("")
                        with container2:
                                st.metric("Total Workers", total_events)
                        # Plot tren dengan warna kuning dan fill di bawah garis
                        fig = px.line(num_events_per_date, x="date", y="num_workers", color_discrete_sequence=['#51b27d'])
                        fig.update_traces(fill='tonexty')  # Mengisi di bawah garis
                        fig.update_xaxes(title_text='', tickfont=dict(size=15))
                        fig.update_yaxes(title_text='')
                        fig.update_layout(height=200)
                        st.plotly_chart(fig, use_container_width=True)                        
                        
                # 2nd ROW
                type_event, category_distribution, total_workload = st.columns([1,0.7,0.8])
                with type_event:
                        df_ = df.drop_duplicates(subset=['act_id'])
                        filtered_df = df_[df_['bulan_tahun'] == month_year_filter]
                        top_5_category_act = filtered_df['category_act'].value_counts().nlargest(5).index
                        filtered_df_top_5 = filtered_df[filtered_df['category_act'].isin(top_5_category_act)]
                        event_counts = filtered_df_top_5.groupby(['type_of_event', 'category_act']).size().reset_index(name='Count')
                        
                        fig = px.bar(event_counts, 
                                x='type_of_event', 
                                y='Count', 
                                color='category_act', 
                                text='Count',  
                                labels={'Count': 'Count', 'category_act': 'Category Act', 'type_of_event': 'Type of Event'},
                                height=500, 
                                width=800)
                        fig.update_traces(textposition='outside', textfont_size=14)  
                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                                                yaxis_title='Count per activity', 
                                                xaxis={'categoryorder':'total descending'},  
                                                showlegend=True,
                                                yaxis=dict(showticklabels=True),
                                                legend=dict(
                                                        orientation='h',
                                                        yanchor='top',
                                                        y=-0.2,
                                                        xanchor='center',
                                                        x=0.5
                                                ))
                        fig.update_xaxes( title_text="",tickfont=dict(size=15))
                        
                        st.subheader("Activity by Event")  
                        st.write("")
                        st.plotly_chart(fig, use_container_width=True)

                        with category_distribution:
                                def plot_pie_chart(data, figure_size=(8, 8), text_size=12):
                                        fig = px.pie(data, names=data.index, values=data.values, 
                                                        labels={'values': 'Count'}, hole=0.4, 
                                                        height=figure_size[1] * 100, width=figure_size[0] * 100)
                                        fig.update_traces(textinfo='percent+label', pull=[0.1] * len(data), textfont_size=text_size)
                                        
                                        # Update layout to place the legend at the bottom
                                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                                                legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=-0.5,  # Adjust as necessary to position the legend below the chart
                                                xanchor="center",
                                                x=0.5
                                                )
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                df_ = df.drop_duplicates(subset=['act_id'])
                                filtered_df = df_[df_['bulan_tahun'] == month_year_filter]
                                category_distribute = filtered_df['kode'].value_counts()
                                st.subheader("Category of Activity Distribution")
                                st.write("")
                                plot_pie_chart(category_distribute, figure_size=(5, 5), text_size=15)
                                
                        with total_workload:
                                def plot_bar_chart_color(data, x_label, y_label):
                                        fig = px.bar(data, x=data.index, y=data.values, labels={'y': y_label})
                                        
                                        # determine the color of the bars based on the highest value
                                        max_value = max(data.values)
                                        colors = ['skyblue' if value != max_value else 'pink' for value in data.values]
                                        
                                        fig.update_traces(marker_color=colors, hoverinfo='y+text', text=data.values, textfont=dict(size=17))
                                        
                                        # update layout to hide y-axis tick labels
                                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                                                xaxis_title=x_label,
                                                xaxis=dict(tickangle=-45),
                                                yaxis=dict(showticklabels=False)
                                        )
                                        fig.update_xaxes(title_text='Week', tickfont=dict(size=15))
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                df_ = df.drop_duplicates(subset=['act_id'])
                                filtered_workload = df_[df_['bulan_tahun'] == month_year_filter]
                                filtered_workload['date'] = pd.to_datetime(filtered_workload['date'])
                                filtered_workload['week_start'] = filtered_workload['date'].dt.to_period('W').apply(lambda r: r.start_time)
                                
                                # group by week start date
                                weekly_activity = filtered_workload.groupby(filtered_workload['week_start'])['activity_description_'].count()
                                st.subheader("Workload by Week")
                                st.write("")
                                plot_bar_chart_color(weekly_activity, 'Week', 'Count Activity')
                
                # 3rd ROW
                used_venue, request_by, work_remarks = st.columns(3)
                with used_venue:
                        def bar_chart_venue(df):
                                fig = px.bar(df,
                                        x='count',
                                        y='venue_event',
                                        orientation='h',
                                        text='count')

                                fig.update_traces(marker=dict(color='skyblue'),
                                                textposition='inside', textfont=dict(size=17))
                                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                                                xaxis_title='count of activity',
                                                yaxis_title='',
                                                xaxis=dict(showgrid=True),
                                                yaxis=dict(categoryorder='total ascending'),
                                                )
                                fig.update_xaxes(title_text='Week', tickfont=dict(size=15))
                                return fig
                        
                        df_ = df.drop_duplicates(subset=["act_id"])
                        filtered_counts = df_[df_['bulan_tahun'] == month_year_filter]
                        venue_counts = filtered_counts['venue_event'].value_counts().reset_index()
                        venue_counts.columns = ['venue_event', 'count']
                        venue_counts = venue_counts.head(5)
                        
                        st.subheader('Most Used Venue')
                        st.write("")
                        st.plotly_chart(bar_chart_venue(venue_counts), use_container_width=True)
                
                with request_by:
                        def bar_chart_request(df):
                                fig = px.bar(df,
                                        x='request_by',
                                        y='count',
                                        text='count')
                                
                                fig.update_traces(marker=dict(color='skyblue'),
                                                textposition='inside', textfont=dict(size=17))
                                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                                                xaxis_title='',
                                                yaxis_title='count of request',
                                                xaxis=dict(categoryorder='total descending')
                                                )
                                fig.update_xaxes(title_text='', tickfont=dict(size=15))
                                return fig

                        df_ = df.drop_duplicates(subset=["act_id"])
                        filtered_counts = df_[df_['bulan_tahun'] == month_year_filter]
                        request_by_counts = filtered_counts['request_by'].value_counts().reset_index()
                        request_by_counts.columns = ['request_by', 'count']
                        request_by_counts = request_by_counts.head(5)
                        
                        st.subheader('Most Request By')
                        st.write("")
                        st.plotly_chart(bar_chart_request(request_by_counts), use_container_width=True)

                with work_remarks:
                        df_ = df.drop_duplicates(subset=["act_id"])
                        filtered_counts = df_[df_['bulan_tahun'] == month_year_filter]
                        counts = filtered_counts['remarks'].value_counts().reset_index()
                        counts.columns = ['status', 'count']
                        total_count = counts['count'].sum()
                        counts['percentage'] = (counts['count'] / total_count)
                        
                        # Membuat pie chart
                        fig = px.pie(counts, 
                                names='status', 
                                values='count', 
                                # title='Work Remarks', 
                                hole=0.5,
                                labels={'status': 'Status', 'count': 'Count'},
                                hover_data={'percentage': ':.0%'},
                                color_discrete_sequence=['skyblue'])

                        # Mengubah layout
                        fig.update_traces(textposition='none', textinfo='none')
                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),showlegend=False)

                        # Menambahkan persentase done di tengah pie chart
                        done_percentage = counts[counts['status'] == 'done']['percentage'].values[0]
                        done_percentage_text = f"{done_percentage:.0%} done"
                        fig.add_annotation(x=0.5, y=0.5, text=f"{done_percentage_text}", 
                                        showarrow=False, font=dict(size=20, color='black'))
                        
                        st.subheader('Work Remarks')
                        st.write("")
                        st.plotly_chart(fig, use_container_width=True)
                
        ## MANPOWER ##
        with manpower:
                filtered_man = df[df['bulan_tahun'] == month_year_filter]
                st.subheader(f'Total Activity vs. Total Employee Daily in :blue[{month_year_filter}]')
                st.write("")
                activity_counts = filtered_man.groupby('date').size().reset_index(name='Total Activity')
                prs_employees = filtered_man[filtered_man['status'] == 'PRS'].groupby('date')['employee_name'].nunique().reset_index(name='Total Employee')
                dual_line_data = pd.merge(activity_counts, prs_employees, on='date', how='outer').fillna(0)

                color_discrete_map = {
                'Total Activity': 'blue',
                'Total Employee': 'orange'
                }

                fig = px.line(
                dual_line_data,
                x='date',
                y=['Total Activity', 'Total Employee'],
                labels={'value': 'Count', 'date': 'Date', 'variable': 'Description'},
                color_discrete_map=color_discrete_map
                )

                fig.update_layout(
                xaxis_title='',
                yaxis_title='Count',
                yaxis=dict(type='log')
                )
                fig.update_xaxes(title_text='', tickfont=dict(size=15))

                st.plotly_chart(fig, use_container_width=True)

                # ACTIVITY HANDLED

                col1, col2 = st.columns(2)
                with col1:
                        df_ = df.drop_duplicates(subset=['id_', 'employee_name'])
                        filtered_man = df_[df_['bulan_tahun'] == month_year_filter]
                        prs_data = filtered_man[filtered_man['status'] == 'PRS']
                        employee_prs_count = prs_data['employee_name'].value_counts().reset_index(name='Count')
                        employee_prs_count.columns = ['Employee', 'Count']

                        employee_prs_count = employee_prs_count.sort_values(by='Count', ascending=False)

                        trace_prs = go.Bar(
                        x=employee_prs_count['Count'], 
                        y=employee_prs_count['Employee'],
                        orientation='h',
                        text=employee_prs_count['Count'],  
                        textposition='auto',  
                        marker_color=['skyblue' if count == employee_prs_count['Count'].max() else 'blue' for count in employee_prs_count['Count']]
                        )

                        layout = go.Layout(
                        title='Total Activity Handled by Employee',
                        xaxis=dict(title='Count', showticklabels=True),  
                        yaxis=dict(title='Employee', showticklabels=True)  
                        )
                        
                        fig_prs = go.Figure(data=[trace_prs], layout=layout)
                        fig_prs.update_xaxes(title_text='Counts', tickfont=dict(size=15))
                        st.plotly_chart(fig_prs, use_container_width=True)

                ## STATUS OFF DAN PRS ##

                with col2:
                        filtered_man['date'] = pd.to_datetime(filtered_man['date'])
                        prs_data = filtered_man[filtered_man['status'] == 'PRS']
                        off_data = filtered_man[filtered_man['status'] == 'OFF']

                        prs_count_weekly = prs_data.groupby('date')['employee_name'].nunique().reset_index(name='Total Employee PRS')
                        off_count_weekly = off_data.groupby('date')['employee_name'].nunique().reset_index(name='Total Employee OFF')

                        trace_prs_weekly = go.Bar(
                        x=prs_count_weekly['date'], 
                        y=prs_count_weekly['Total Employee PRS'],
                        text=prs_count_weekly['Total Employee PRS'],
                        textposition='auto', 
                        name='PRS', 
                        marker_color='orange'
                        )
                        
                        trace_off_weekly = go.Bar(
                        x=off_count_weekly['date'], 
                        y=off_count_weekly['Total Employee OFF'],
                        text=off_count_weekly['Total Employee OFF'], 
                        textposition='auto', 
                        name='OFF', 
                        marker_color='red'
                        )

                        layout_weekly = go.Layout(
                        title='Total Present and Off Employees Weekly',
                        xaxis=dict(title='Day', tickformat='%Y-%m-%d'),
                        yaxis=dict(title='Count', showticklabels=False), 
                        barmode='group'
                        )

                        fig_status_weekly = go.Figure(data=[trace_prs_weekly, trace_off_weekly], layout=layout_weekly)
                        fig_status_weekly.update_xaxes(title_text='Days', tickfont=dict(size=15))
                        st.plotly_chart(fig_status_weekly, use_container_width=True)

        ## WORKLOAD DETAIL ##
        with workload_detail:

                initialize_state()

                df = pd.read_excel('data/gitfile2 (1).xlsx')
                df = df.drop_duplicates(subset="act_id")

                results = render_plotly_ui(df)
                df = results['df']
                top_categories_ = results['top_categories_']
                current_query_result = results['current_query']
                month_year_filter = results['month_year_filter']
                update_state(current_query_result)

                # Tampilkan data yang dipilih oleh pengguna
                selected_data = display_selected_data()

                if selected_data is not None and len(selected_data) > 0:
                    first_item = selected_data[0]
                    filtered_rows = df[df['id_'] == first_item]
                    objek_filter = filtered_rows['category_act'].values[0]
                else:
                    objek_filter = "perbaikan"

                df = pd.read_excel('data/gitfile2 (1).xlsx')
                df = df.drop_duplicates(subset="act_id")
                df_ = df[['objek', 'category_act', 'act_id', 'area', 'venue_act']]
                filtered_df_ = pd.merge(top_categories_, df_, on="category_act", how='left')
                filtered_df_ = filtered_df_.drop_duplicates(subset='act_id')
                filtered_rows = filtered_df_[filtered_df_['category_act'] == objek_filter]

                # CSS for styling
                css = """
                <style>
                .container {
                    margin: 20px 0;
                }
                .plotly-chart {
                    margin-top: -30px !important;
                    height: 600px !important; /* Adjust height as needed */
                }
                .matplotlib-plot {
                    height: 600px !important;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                </style>
                """

                # Insert the CSS into the app using markdown
                st.markdown(css, unsafe_allow_html=True)

                container1, container2 = st.columns(2)
                # Container 1: WordCloud
                with container1:
                    st.subheader(f"WordCloud of :orange[{objek_filter.title()}]")
                    filtered_df_ = filtered_df_[filtered_df_['category_act'] == objek_filter]
                    text = ' '.join(filtered_df_['objek'].dropna())
                    tokens = word_tokenize(text)
                    stop_words = set(stopwords.words('indonesian'))
                    conjunctions = ["untuk", "atau", "tetapi", "maintenance", "konek", "perbaikan", "pengkonekan", "basement", "instalasi", "loading dock", 'area', 'toilet', 'area', "lepas", "ruang", "snack bar", "event", "snack", "pemasangan", 'unit', 'bersih', 'name', 'pengecatan', 'sisi', 'wudhu', 'ex']
                    stop_words.update(conjunctions)

                    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]

                    word_freq = Counter(filtered_tokens)
                    wordcloud = WordCloud(background_color='white', max_words=100, stopwords=stop_words).generate_from_frequencies(word_freq)

                    plt.figure(figsize=(10, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')

                    st.pyplot(plt, use_container_width=True)

                # Container 2: Treemap
                with container2:
                    st.subheader(f"Mapping Area Activity of :orange[{objek_filter.title()}]")
                    filtered_df_ = filtered_df_.drop(columns=['num_events'])
                    num_events_per_date = filtered_df_.groupby(['area', 'venue_act'])['act_id'].count().reset_index(name='num_events')

                    fig = px.treemap(num_events_per_date, path=["area", "venue_act"], values="num_events", color="venue_act")
                    fig.update_layout(margin=dict(t=0))

                    st.plotly_chart(fig, use_container_width=True, use_container_height=True)

st.markdown('-------')
