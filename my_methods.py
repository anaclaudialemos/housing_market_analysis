import streamlit      as st
import pandas         as pd
import numpy          as np
import plotly_express as px

import folium
import geopandas

from streamlit_option_menu import option_menu
from streamlit_folium      import folium_static
from folium.plugins        import MarkerCluster
from PIL                   import Image


@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)

    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def display_data_overview(data, f_attributes, f_zipcode):

    if (f_attributes != []) & (f_zipcode != []):
        to_disp = list(f_attributes)
        to_disp.append('id')
        return data.loc[data['zipcode'].isin(f_zipcode), to_disp[::-1]]

    elif (f_attributes != []) & (f_zipcode == []):
        to_disp = list(f_attributes)
        to_disp.append('id')
        return data[to_disp[::-1]]
        
    elif (f_attributes == []) & (f_zipcode != []):
        return data.loc[data['zipcode'].isin(f_zipcode), :]
    
    else:
        return data

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def display_data_by_zipcode(data, f_zipcode):

    count      = data[['id', 'zipcode']].groupby('zipcode').count()
    price      = data[['buying_price', 'zipcode']].groupby('zipcode').mean()
    sqft       = data[['sqft_living', 'zipcode']].groupby('zipcode').mean()
    price_sqft = data[['price_sqft', 'zipcode']].groupby('zipcode').mean().reset_index()

    data = pd.merge(count, price,     on='zipcode', how='inner')
    data = pd.merge(data, sqft,       on='zipcode', how='inner')
    data = pd.merge(data, price_sqft, on='zipcode', how='inner')

    data.columns = ['Zipcode', 'Total Properties', 'Buying Price', 'Sqft Living', 'Price/Sqft']

    if f_zipcode == []:
        return (data, 225)

    else:
        return (data[data['Zipcode'].isin(f_zipcode)], None)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def display_descriptive_qnt(data, f_attributes):

    desc_qnt = data.describe().T.drop(columns='count')
    desc_qnt = desc_qnt.rename(columns={'mean': 'Mean', 'std': 'Std Dev', 'min': 'Min', '50%': 'Median', 'max': 'Max'})

    if f_attributes == []:
        return (desc_qnt, 225)

    else:
        desc_qnt = desc_qnt.loc[f_attributes]
        return (desc_qnt, None)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def display_descriptive_qlt(data, f_attributes):

    min_value = pd.DataFrame(data.apply(np.min)).rename(columns = {0: 'Min'}).astype(str)
    max_value = pd.DataFrame(data.apply(np.max)).rename(columns = {0: 'Max'}).astype(str)

    desc_qlt = data.astype('category').describe().T.drop(columns = 'count').rename(columns = {'unique': 'Uniques', 'top': 'Mode', 'freq': 'Mode Freq'}).astype(str)
    desc_qlt = pd.concat([min_value, max_value, desc_qlt], axis=1)

    if f_attributes == []:
        return (desc_qlt, 225)

    else:
        desc_qlt = desc_qlt.loc[f_attributes]
        return (desc_qlt, None)

def get_filtered_data(data):

    min_price          = int(data['buying_price'].min())
    max_price          = int(data['buying_price'].max())
    f_price            = st.sidebar.slider('Price range:', min_price, max_price, (min_price, max_price))
    to_disp_price      = data.loc[data['buying_price'].between(f_price[0], f_price[1])]

    if 'expected_profit' in data.columns:

        min_profit     = int(data['expected_profit'].min())
        max_profit     = int(data['expected_profit'].max())
        f_profit       = st.sidebar.slider('Expected profit range:', min_profit, max_profit , (min_profit, max_profit))
        to_disp_profit = data.loc[data['expected_profit'].between(f_profit[0], f_profit[1])]

    data['date']       = pd.to_datetime(data['date']).apply(lambda x: x.date())
    min_date           = data['date'].min()
    max_date           = data['date'].max()
    f_date             = st.sidebar.slider('Date range:', min_date, max_date, (min_date, max_date))
    to_disp_date       = data.loc[data['date'].between(f_date[0], f_date[1])]

    min_yr_built       = int(data['yr_built'].min())
    max_yr_built       = int(data['yr_built'].max())
    f_yr_built         = st.sidebar.slider('Year of built range:', min_yr_built, max_yr_built, (min_yr_built, max_yr_built))
    to_disp_yr_built   = data.loc[data['yr_built'].between(f_yr_built[0], f_yr_built[1])]

    cond_opt           = sorted(data['condition'].unique().tolist())   
    f_cond             = st.sidebar.selectbox('Min condition:', cond_opt)
    to_disp_cond       = data.loc[data['condition'] >= f_cond]

    bed_opt            = sorted(data['bedrooms'].unique().tolist())
    f_bed              = st.sidebar.selectbox('Min number of bedrooms:', bed_opt)
    to_disp_bed        = data.loc[data['bedrooms'] >= f_bed]
    
    bath_opt           = sorted(data['bathrooms'].unique().tolist())                   
    f_bath             = st.sidebar.selectbox('Min number of bathrooms:', bath_opt)
    to_disp_bath       = data.loc[data['bathrooms'] >= f_bath]

    floors_opt         = sorted(data['floors'].unique().tolist())              
    f_floors           = st.sidebar.selectbox('Min number of floors:', floors_opt)
    to_disp_floors     = data.loc[data['floors'] >= f_floors]


    waterf_opt = ['Show All', 'Waterfront', 'No Waterfront']
    f_waterf = st.sidebar.radio('Waterfront options:', waterf_opt)

    if f_waterf == 'Show All':
        to_disp_waterf = data
    
    elif f_waterf == 'Waterfront':
        to_disp_waterf = data.loc[data['waterfront'] == 1]

    else:
        to_disp_waterf = data.loc[data['waterfront'] == 0]
    
    view_opt = ['Show All', 'Good View', 'No Good View']
    f_view = st.sidebar.radio('View options:', view_opt)

    if f_view == 'Show All':
        to_disp_view = data

    elif f_view == 'Good View':
        to_disp_view = data.loc[data['view'] >= 3]

    else:
        to_disp_view = data.loc[data['view'] < 3]


    basem_opt = ['Show All', 'With Basement', 'No Basement']
    f_basem = st.sidebar.radio('Basement options:', basem_opt)

    if f_basem == 'Show All':
        to_disp_basem = data

    elif f_basem == 'Basement':
        to_disp_basem = data.loc[data['sqft_basement'] > 0]

    else:
        to_disp_basem = data.loc[data['sqft_basement'] == 0]

    min_sqft_living   = int(data['sqft_living'].min())
    max_sqft_living   = int(data['sqft_living'].max())
    f_sqft_livg       = st.sidebar.slider('Living area (sqft) range:', min_sqft_living, max_sqft_living, (min_sqft_living, max_sqft_living))
    to_disp_sqft_livg = data.loc[data['sqft_living'].between(f_sqft_livg[0], f_sqft_livg[1])]

    min_sqft_lot      = int(data['sqft_lot'].min())
    max_sqft_lot      = int(data['sqft_lot'].max())
    f_sqft_lot        = st.sidebar.slider('Lot area (sqft) range:', min_sqft_lot, max_sqft_lot, (min_sqft_lot, max_sqft_lot))
    to_disp_sqft_lot  = data.loc[data['sqft_lot'].between(f_sqft_lot[0], f_sqft_lot[1])]

    if 'expected_profit' in data.columns:

        to_disp = pd.merge(to_disp_price, to_disp_date,      how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_yr_built,  how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_cond,      how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_bed,       how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_bath,      how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_floors,    how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_waterf,    how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_view,      how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_basem,     how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_sqft_livg, how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_sqft_lot,  how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_profit,    how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        
        to_disp.drop([i for i in to_disp.columns if 'remove' in i], axis=1, inplace=True)

        return to_disp, f_price, f_date, f_yr_built, f_cond, f_bed, f_bath, f_floors, f_waterf, f_view, f_basem, f_sqft_livg, f_sqft_lot, f_profit

    else:

        to_disp = pd.merge(to_disp_price, to_disp_date,      how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_yr_built,  how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_cond,      how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_bed,       how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_bath,      how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_floors,    how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_waterf,    how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_view,      how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_basem,     how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_sqft_livg, how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        to_disp = pd.merge(to_disp,       to_disp_sqft_lot,  how='inner', left_index=True, right_index=True, suffixes=('', '_remove'))
        
        to_disp.drop([i for i in to_disp.columns if 'remove' in i], axis=1, inplace=True)

        return to_disp, f_price, f_date, f_yr_built, f_cond, f_bed, f_bath, f_floors, f_waterf, f_view, f_basem, f_sqft_livg, f_sqft_lot


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def display_portfolio_density_map(data):

    map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)
    marker_cluster = MarkerCluster().add_to(map)

    for name, row in data.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold {0} USD on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                          row['buying_price'], row['date'], row['sqft_living'], row['bedrooms'], row['bathrooms'],
                          row['yr_built'])).add_to(marker_cluster)

    return map

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def display_price_density_map(data, geodata):

    df_map = data[['buying_price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df_map.columns = ['Zipcode', 'Price']

    geodata = geodata[geodata['ZIP'].isin(df_map['Zipcode'].tolist())]

    map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)
    folium.Choropleth(data=df_map, geo_data=geodata, columns=['Zipcode', 'Price'], key_on='feature.properties.ZIP',
                      fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2, legend_name='Average Price').add_to(map)

    return map

def display_price_dist(data):

    st.sidebar.subheader('Prices Distribution')
    f_zipcode  = st.sidebar.multiselect('Zipcodes:', options=data['zipcode'].sort_values().unique())
    data_price = data.loc[data['zipcode'].isin(f_zipcode), ['buying_price', 'zipcode']] if sum(data['zipcode'].isin(f_zipcode)) > 0 \
                                                                                        else   data[['buying_price', 'zipcode']].copy()

    min_price = int(data_price['buying_price'].min())
    max_price = int(data_price['buying_price'].max())

    f_price    = st.sidebar.slider('Price range:', min_price, max_price, (min_price, max_price))
    data_price = data_price[data_price['buying_price'].between(f_price[0], f_price[1])]

    price_plot = px.histogram(data_price, x='buying_price', color='zipcode' if f_zipcode != [] else None, marginal='box', 
                              labels={'buying_price': 'Price (USD)'})
    
    st.subheader('Prices Distribution')
    st.markdown(' An histogram showing how prices are distributed in the dataset. You can adjust the horizontal axis range using' 
                ' the commercial options in the sidebar.')

    st.plotly_chart(price_plot, use_container_width=True)
    
    return None

def display_price_date(data):

    st.sidebar.subheader('Daily Price Evolution')

    data['date'] = pd.to_datetime(data['date']).apply(lambda x: x.date())
    min_date = data['date'].min()
    max_date = data['date'].max()

    f_date = st.sidebar.slider('Date range:', min_date, max_date, (min_date, max_date))

    data_dt = data[data['date'].between(f_date[0], f_date[1])]
    data_dt = data_dt.groupby('date').mean().reset_index()

    date_plot = px.line(data_dt, x='date', y='buying_price', 
                        labels={'buying_price': 'Average Price (USD)', 'date': 'Date'})

    st.subheader('Daily Price Variations')
    st.markdown(' Daily price variation according to the date of registration. You can filter the horizontal axis range using the' 
                ' commercial options in the sidebar.')

    st.plotly_chart(date_plot, use_container_width=True)

    return None

def display_price_yrbuilt(data):

    st.sidebar.subheader('Average Price per Year Built')

    min_yr_built = int(data['yr_built'].min())
    max_yr_built = int(data['yr_built'].max())

    f_yr_built = st.sidebar.slider('Year of built range:', min_yr_built, max_yr_built, (min_yr_built, max_yr_built))

    data_yr = data[data['yr_built'].between(f_yr_built[0], f_yr_built[1])]
    data_yr = data_yr.groupby('yr_built').mean().reset_index()

    yr_built_plot = px.line(data_yr, x='yr_built', y='buying_price',
                            labels={'buying_price': 'Average Price (USD)', 'yr_built': 'Year of Built'})

    st.subheader('Average Price per Year Built')
    st.markdown(' How prices vary according to the year properties were built. You can filter the horizontal axis range using the' 
                ' commercial options in the sidebar.')

    st.plotly_chart(yr_built_plot, use_container_width=True)

    return None

def display_physics_optios():
    phys_opt = st.sidebar.radio('Select one of the following options to display attributes plots:',
                                options=('Properties by Bedrooms',
                                         'Properties by Bathrooms',
                                         'Properties by Floors',
                                         'Properties by Waterfront',
                                         'Properties by View',
                                         'Show all'), index=5)
    
    return phys_opt

def display_attributes_hist(data, col, nbins=40):

    data = data[col]

    fig = px.histogram(data, x=col, nbins=nbins)
    st.plotly_chart(fig, use_container_width=True)

    return None

def display_price_waterfront(data):

    data_waterfront = data[['buying_price', 'waterfront']].groupby('waterfront').median().reset_index()

    price_plot = px.bar(data_waterfront, x='waterfront', y='buying_price',
                        labels={'buying_price': 'Median Price (USD)', 'waterfront': 'Properties'})

    price_plot.update_layout(xaxis=dict(tickmode='array',
                                        tickvals=[0, 1],
                                        ticktext=['Not Waterfront', 'Waterfront']))

    isnot_waterfront = data_waterfront.iloc[0,1]
    is_waterfront = data_waterfront.iloc[1,1]

    perc_diff= abs(((is_waterfront - isnot_waterfront) / isnot_waterfront)*100)
    perc_diff= str(round(perc_diff, 2))

    return price_plot, perc_diff


def display_price_view_condition(data):
    
    gv_notgc = data[(data['view'] >= 3) & (data['condition'] <= 3)][['buying_price']].rename(columns={"buying_price": "gv_notgc"}).median()
    notgv_gc = data[(data['view']  < 3) & (data['condition']  > 3)][['buying_price']].rename(columns={"buying_price": "notgv_gc"}).median()

    data_view_condition = pd.concat([gv_notgc, notgv_gc]).reset_index()
    data_view_condition.columns = ['view_condition','buying_price']

    price_plot = px.bar(data_view_condition, x='view_condition', y='buying_price',
                        labels={'buying_price': 'Median Price (USD)', 'view_condition': 'Properties'})

    price_plot.update_layout(xaxis=dict(tickmode='array',
                                        tickvals=[0, 1],
                                        ticktext=['Good View and Not Good Condition', 'Not Good View and Good Condition']))
    
    gv_notgc = data_view_condition.iloc[0,1]
    notgv_gc = data_view_condition.iloc[1,1]

    perc_diff = abs(((gv_notgc - notgv_gc) / notgv_gc)*100)
    perc_diff = str(round(perc_diff, 2))

    return price_plot, perc_diff

def display_price_age(data):

    data_newer = data[['buying_price', 'newer']].groupby('newer').median().reset_index()

    newer_plot = px.bar(data_newer, x='newer', y='buying_price', 
                        labels={'buying_price': 'Average Price (USD)', 'newer': 'Properties'})
    
    newer_plot.update_layout(xaxis=dict(tickmode='array', 
                                        tickvals=[0, 1],
                                        ticktext=['Older', 'Newer']))

    isnot_newer = data_newer.iloc[0,1]
    is_newer    = data_newer.iloc[1,1]

    perc_diff = abs(((is_newer  - isnot_newer) / isnot_newer)*100)
    perc_diff = str(round(perc_diff, 2))

    return newer_plot, perc_diff

def display_living_space(data):

    data_living_space = data[['yr_built', 'sqft_living']].groupby('yr_built').median().reset_index()

    living_space_plot = px.line(data_living_space, x='yr_built', y='sqft_living',
                                labels={'yr_built': 'Year of Built', 'sqft_living': 'Median of Living Space (sqft)'})

    st.plotly_chart(living_space_plot, use_container_width=True)

    return None

def display_price_year_season(data):
    
    data_season = data[['date','buying_price', 'season']]
    data_season['date'] = pd.to_datetime(data_season['date']).apply(lambda x: x.date())
    data_season['year'] =data_season['date'].apply( lambda x: x.year )

    data_season = data_season[['buying_price', 'season', 'year']].groupby(['season', 'year']).mean().reset_index()

    season_plot = px.bar(data_season, x='year', y='buying_price', color='season', barmode='group',
                         labels={'season': '', 'year':'', 'buying_price': 'Average Price (USD)'})
    
    st.plotly_chart(season_plot, use_container_width=True)

    return None

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def display_suggestions_tobuy(data):

    median_price_zipcode = data[['buying_price', 'zipcode']].groupby('zipcode').median().reset_index().rename(
                           columns={'buying_price': 'median_price_zipcode'})
    
    data = pd.merge(data, median_price_zipcode, on='zipcode', how='inner')

    data['decision'] = data[['buying_price', 
                              'median_price_zipcode', 
                              'waterfront', 
                              'view', 
                              'condition', 
                              'newer']].apply(
                                  lambda x: 1 if ((x['buying_price'] <= x['median_price_zipcode']) & (x['waterfront'] == 1) | 
                                                  (x['buying_price'] <= x['median_price_zipcode']) & (x['view'] >= 3) & (x['condition'] >= 2) | 
                                                  (x['buying_price'] <= x['median_price_zipcode']) & (x['condition'] >= 3) & (x['newer'] == 1) |
                                                  (x['buying_price'] <= x['median_price_zipcode']) & (x['condition'] >= 4)) else 0, axis=1)
    
    df_tobuy = data[data['decision'] == 1].copy().drop(columns='decision')

    df_tobuy['maximal_expend'] = df_tobuy[['buying_price', 
                                           'condition']].apply(
                                               lambda x: x['buying_price'] * 0.12 if x['condition'] <= 2 else
                                                         x['buying_price'] * 0.08 if x['condition'] == 3 else
                                                         x['buying_price'] * 0.06, axis=1)

    df_tobuy['suggested_selling_price'] = df_tobuy[['buying_price',
                                                     'median_price_zipcode',
                                                     'maximal_expend',
                                                     'waterfront',
                                                     'view']].apply(
                                                         lambda x: x['buying_price'] * 1.40 if   x['waterfront'] == 1 else 
                                                                   x['buying_price'] * 1.35 if ((x['waterfront'] == 0) & (x['view'] >= 3)) else 
                                                                   x['buying_price'] * 1.30 if ((x['waterfront'] == 0) & (x['view'] <  3) & 
                                                                                               ((x['buying_price'] + x['maximal_expend']) < x['median_price_zipcode'])) else 
                                                                   x['buying_price'] * 1.25, axis=1)

    df_tobuy['expected_profit'] = df_tobuy[['buying_price', 
                                            'maximal_expend', 
                                            'suggested_selling_price']].apply(
                                                lambda x: x['suggested_selling_price'] - x['maximal_expend'] - x['buying_price'], axis=1)

    tot_initial_investment = np.round(df_tobuy['buying_price'].sum(), 2)
    tot_maximal_expend     = np.round(df_tobuy['maximal_expend'].sum(), 2)
    tot_investment         = tot_initial_investment + tot_maximal_expend
    tot_expected_profit    = np.round(df_tobuy['expected_profit'].sum(), 2)
    percentage             = np.round((tot_expected_profit / tot_investment) * 100, 2)

    return (df_tobuy, tot_initial_investment, tot_maximal_expend, tot_investment, tot_expected_profit, percentage)
