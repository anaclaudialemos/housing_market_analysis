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

## pre-defined functions 
from my_methods import*

## --- streamlit settings

## layout config
st.set_page_config(layout='wide')
pd.set_option('display.float_format', '{:.2f}'.format)

## map size config
make_maps_responsive= """ <style> [title~="st.iframe"] { width: 100%} </style> """
st.markdown(make_maps_responsive, unsafe_allow_html=True)


## --- execution 

if __name__ == "__main__":

    ## data extraction
    df = get_data('datasets/kc_house_data_processed.csv')	
    geofile = get_geofile( 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson' )

    df = df[['id', 'date', 'buying_price', 'season', 'zipcode', 'newer', 'yr_built', 'bedrooms', 'bathrooms', 'sqft_living', 'price_sqft' ,'sqft_lot', 'floors',
             'basement', 'waterfront', 'view', 'condition', 'grade', 'renovated', 'yr_renovated','sqft_above', 'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]
    
    ## sidebar image settings
    image=Image.open('images/sidebar.png')
    st.sidebar.image(image, use_column_width=True)
    
    options = option_menu(None, ['Overview', 'Maps', 'Statistics', 'DataViz', 'Insights', 'Opportunities'], orientation='horizontal',
                          menu_icon='menu-up', icons=['table', 'map', 'search', 'bar-chart', 'lightbulb', 'currency-dollar'], 
                          styles={'nav-link': {'font-size': "14px"}, 'nav-link-selected': {'font-size': '14px', 'background-color': '#242b64'}})

    if options == 'Overview':

        st.title('King County Housing Dashboard')
        st.markdown(' The dataset consists of residential property prices from King County an area in the US State of Washington, whose seat and' 
                    ' most populous city is Seattle. Below you can have an overview of how the dataset looks like. Use the sidebar to see the' 
                    ' interactive options.')
        st.markdown('\n' )

        st.markdown(' This dashboard is part of a complete business intelligence project concerned with profiting from buying and selling properties.'
                    ' Click [here](https://github.com/anaclaudialemos/buying_selling_houses) to get to the Github repository.' )
        st.markdown('\n' )
       
        st.markdown('**Disclaimer:** The data used in this application are public and were taken from '
                    '[Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction). This is a fictional project, developed for learning purposes.' 
                    ' The context and observed trends may not represent real-life behavior.')
        st.markdown('Illustration on sidebar by [Julia Gnedin](https://icons8.com/illustrations/author/627444) from [Ouch!](https://icons8.com/illustrations)')
        st.markdown('---')

        ## data overview
        st.sidebar.header('Portfolio Overview Options')
        f_attributes = st.sidebar.multiselect('Attributes', options=df.columns.drop('id'))
        f_zipcode = st.sidebar.multiselect('Zipcodes (to filter by zipcodes):', df['zipcode'].sort_values().unique())

        st.header('Portfolio Overview')
        st.dataframe(display_data_overview(df, f_attributes, f_zipcode))

        # downloading table
        df_csv = convert_csv(display_data_overview(df, f_attributes, f_zipcode))
        st.download_button(label="Download this table as CSV",
                           data=df_csv, file_name='selected_attributes.csv',
                           mime='text/csv')
    
    if options == 'Maps':
        st.title('Maps')
        st.markdown('This section is intended to display geographical data. Regions are separated by zip code. The run may took a while.' )
        st.markdown('---')
         
        st.header('Portfolio Density')
        st.sidebar.header('Portfolio Density Options')

        ## filters and filtered data to portfolio density
        filtered_df, f_price, f_date, f_ybuilt, f_cond, f_bed, f_bath, f_floors, f_waterf, f_view, f_basem, f_living, f_lot = get_filtered_data(df)
        folium_static(display_portfolio_density_map(filtered_df))

        st.header('Prices Density by Zipcode')
        folium_static(display_price_density_map(df, geofile))

    if options == 'Statistics':

        st.title('Descriptive Statistics')
        st.markdown(' The most important metrics and statistics are shown in this section.'
                    ' The main goal here is presenting a descriptive analysis statistics of the data in a simple way.'
                    ' If you want view the portfolio, check the box Display Portfolio Overview, on sidebar.')
        st.markdown('---')

        if  st.sidebar.checkbox('Display Portfolio Overview'):
            st.sidebar.header('Portfolio Overview Options')
            f_attributes = st.sidebar.multiselect('Attributes', options=df.columns.drop('id'))
            f_zipcode = st.sidebar.multiselect('Zipcodes (to filter by zipcodes):', df['zipcode'].unique())

            st.header('Portfolio Overview')
            st.dataframe(display_data_overview( df, f_attributes, f_zipcode))
        
        ## statistics
        st.sidebar.header('Values by Zipcode Options')
        st.header('Values by Zipcode')
        st.markdown('In the table below you can see averaged values for price, sqft_living, and price/sqft on each region (labelled by zipcode). Use sidebar options to filter.')
        st.markdown( '\n' )

        f_zipcode = st.sidebar.multiselect('Select zipcodes to display', options=df['zipcode'].unique())
        
        to_disp, h = display_data_by_zipcode( df,f_zipcode )
        st.dataframe(to_disp, height=h)

        st.sidebar.header('Descriptive Statistics Options')
        st.header('Descriptive Statistics')
        st.markdown('This tables shows descriptive statistics for several attributes. You can control which attribute and metric are shown by using the options in the sidebars.' )
        st.markdown('\n')

        c1, c2 = st.columns((1.5, 1))

        c1.markdown('**Quantitatives Attributes**')
        
        df_qnt = df[['buying_price', 'bedrooms', 'bathrooms', 'floors', 'sqft_living', 'price_sqft', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']]
        f_qnt_attributes = st.sidebar.multiselect('Select quantitatives attributes to describe', options=df_qnt.columns)
        
        to_disp, h = display_descriptive_qnt(df_qnt, f_qnt_attributes)
        c1.dataframe(to_disp, height=h)

        c2.markdown('**Qualitative Attributes**')

        df_qlt = df[['date', 'zipcode', 'yr_built', 'yr_renovated', 'waterfront', 'view', 'condition', 'grade', 'basement', 'newer', 'season']]
        f_qlt_attributes = st.sidebar.multiselect('Select qualitatives attributes to describe', options=df_qlt.columns)
        
        to_disp, h = display_descriptive_qlt(df_qlt, f_qlt_attributes)
        c2.dataframe(to_disp, height=h)
    
    if options == 'DataViz':

        st.title('Data Visualization')
        st.markdown('This section is intended to view the distribution of the most important attributes of the portfolio.')
        st.markdown('---')

        ## commercial attributes
        st.header('Commercial Attributes')
        st.sidebar.title('Commercial Options')
        st.markdown('In this section, you can see figures related to commercial attributes. Use the commercial options in the sidebar to filter these data.' )

        switcher = {'Prices Distribution'         : display_price_dist,
				    'Daily Price Variations'      : display_price_date,
				    'Average Price per Year Built': display_price_yrbuilt}

        com_opt = st.sidebar.radio('Select one of the following options to display commercial plots:',
                          options=('Prices Distribution', 
                                   'Daily Price Variations',
                                   'Average Price per Year Built',
                                   'Show all'), index=3)
  
        if com_opt in switcher.keys():
            try:
                for call in switcher[com_opt]:
                    call(df)
            except:
                switcher[com_opt](df)
        else:
            switcher['Prices Distribution'](df)
            switcher['Daily Price Variations'](df)
            switcher['Average Price per Year Built'](df)
        
	    ## physical attributes
        st.header('Physical Attributes')
        st.sidebar.title('Physical Options')
        st.markdown('Below you can see histograms showing the distribution of the houses according to the selected attributes. '
                    'Use the sidebar to choose for displaying either one histogram or all available histograms.')

        switcher = {'Properties by Bedrooms'  : 'bedrooms',
                    'Properties by Bathrooms' : 'bathrooms',
                    'Properties by Floors'    : 'floors',
                    'Properties by Waterfront': 'waterfront',
                    'Properties by View'      : 'view'}

        phys_opt = display_physics_optios()
        
        if phys_opt in switcher.keys():
            st.subheader('Properties by '+switcher[phys_opt].capitalize())
            display_attributes_hist(df,   switcher[phys_opt])
        else:
            st.subheader('Properties by '+switcher['Properties by Bedrooms'].capitalize())
            display_attributes_hist(df,   switcher['Properties by Bedrooms'])

            st.subheader('Properties by '+switcher['Properties by Bathrooms'].capitalize())
            display_attributes_hist(df,   switcher['Properties by Bathrooms'])

            st.subheader('Properties by '+switcher['Properties by Floors'].capitalize())
            display_attributes_hist(df,   switcher['Properties by Floors'])

            st.subheader('Properties by '+switcher['Properties by Waterfront'].capitalize())
            display_attributes_hist(df,   switcher['Properties by Waterfront'])

            st.subheader('Properties by '+switcher['Properties by View'].capitalize())
            display_attributes_hist(df,   switcher['Properties by View'])

    if options == 'Insights':

        st.title('Main Insights')
        st.markdown('This section is intended to present the main business insights that come out of the exploratory data analysis.')
        st.markdown('---')

        ## price by waterfront
        price_plot_waterfront, diff_perc_waterfront = display_price_waterfront(df)

        st.subheader(f'At the median, waterfront properties are {diff_perc_waterfront}% more expensive.')

        st.markdown(' It was to be expected, considering the same region, that waterfront properties would cost more than without it.'
                    ' What was not expected was that the difference would be so big. Having the chance to buy waterfront, property'
                    ' for a low price allows you to practice a higher profit margin.')

        st.plotly_chart(price_plot_waterfront, use_container_width=True)

        st.markdown(' **Business Action Suggestion:** A waterfront property, sold at a price lower than the regional median, should'
                    ' be purchased regardless of the condition.')

        ## price by view and condition
        price_plot_view_cond, diff_perc_view_cond =  display_price_view_condition(df)

        st.subheader(f' At the median, properties with a good view that are not in good condition are {diff_perc_view_cond}% more expensive,'
                      ' than properties that do not have good view that are in good condition.')
        st.markdown(' Having a good view makes the property much more expensive than its condition. Having the chance to buy a property with'
                    ' a good view, for a low price allows you to practice a higher profit margin, regardless of the propertie condition.')

        st.plotly_chart(price_plot_view_cond, use_container_width=True)

        st.markdown(' **Business Action Suggestion:** A property, which has a good view (from 3 upwards), being sold for a price lower than'
                    ' the regional median price, should be purchased if the if the condition is from 2 upwards.')

        ## price by properties age
        price_plot_age, diff_perc_age = display_price_age(df)

        st.subheader(f'On average, older properties are only {diff_perc_age}% cheaper than newer.')
        st.markdown(' Older properties (built before 1965) were expected to cost at least 20% less than newer properties.' 
                    ' Counterintuitively, as noted in the graph, older properties are, on median, only about 6% cheaper.')

        st.plotly_chart(price_plot_age, use_container_width=True)

        st.markdown(' **Business Action Suggestion:** It is more appropriate to choose newer properties over older ones. The small price ' 
                    ' difference may not compensate for possible renovation expenses in the case of older properties. Therefore, when thinking'
                    ' about buying older properties, we should choose those in very good condition (from 4 upwards).')

        ## living space over the years
        st.subheader('The living space of the properties increased over construction year timeline.')
        st.markdown(' This shows that people nowadays have a preference for bigger properties. This preference reinforces the proposal to buy newer' 
                    ' properties.') 
        
        display_living_space(df)

        st.markdown(' **Business Action Suggestion:** Preferably to buy newer properties. They are generally larger, which people prefer, and ' 
                    ' typically, cost less to renovate newer properties.')

        ## price by season of year
        st.subheader('It cannot be said that there is a better season to sell a property.')

        st.markdown(' Prices are slightly higher in spring and summer, but when taking into account the relative deviation from one average' 
                    ' price to another, we can not say that there is better season to sell a property, more data is needed.')

        display_price_year_season(df)

        st.markdown('**Business Action Suggestion:** With the above consideration, selling a property as quickly as possible is better for increasing net'
                    '  working capital.')
        

    if options == 'Opportunities':

        ## final dataframe
        st.header('Business Opportunities')
        st.markdown(' Below we can see a table indicating properties met as potential business according to the pre-defined assumptions.' 
                    ' If you want to see the map with potential business properties check the box Display Map on sidebar.')
        st.markdown('---')
        
        df_tobuy, tot_initial_investment, tot_maximal_expend, tot_investment, tot_expected_profit, percentage = display_suggestions_tobuy(df)

        ## data overview
        st.sidebar.header('Business Opportunities Options')
        f_attributes = st.sidebar.multiselect('Select attributes', options=df_tobuy.columns.drop('id'))
        f_zipcode = st.sidebar.multiselect('Select the zipcode (to filter by zipcode):', df_tobuy['zipcode'].sort_values().unique())

        st.sidebar.markdown('\n')
        st.dataframe(display_data_overview(df_tobuy, f_attributes, f_zipcode))

        # downloading table
        df_csv = convert_csv(display_data_overview(df_tobuy, f_attributes, f_zipcode))
        st.download_button(label="Download this table as CSV",
                           data=df_csv, file_name='selected_attributes_suggestions_to_buy.csv',
                           mime='text/csv')
        
        st.markdown('\n')

        st.markdown(f'**Total Profit: {tot_expected_profit:,.2f} USD, which represents {percentage}% of the total investment.**')
        st.markdown(f"""Investment in Buying Properties: {tot_initial_investment:,.2f} USD   
                        Investment in Repairs and Renovations: {tot_maximal_expend:,.2f} USD""")

        if st.sidebar.checkbox('Display Map'):
            
            ## filters and filtered data to map
            filtered_df, f_price, f_date, f_ybuilt, f_cond, f_bed, f_bath, f_floors, f_waterf, f_view, f_basem, f_living, f_lot, f_profit = get_filtered_data(df_tobuy)

            folium_static(display_portfolio_density_map(filtered_df))
    
    st.markdown( '---' )
    st.markdown(" Made by Ana Cláudia Lemos, as part of a complete business intelligence project, which you can find on [GitHub](https://github.com/anaclaudialemos/buying_selling_houses).")
    st.markdown(" 🤝 Connect with me! Here is my [Linkedin](https://linkedin.com/in/anaclaudiarlemos).")
    