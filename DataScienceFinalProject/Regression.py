import numpy as np
from PlasticWaste import final_df 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Normalize to remove all commas and 
final_df['GDP'] = final_df['GDP'].replace({'\$': '',}, regex=True);
final_df = final_df.replace({',': ''}, regex=True);
final_df = final_df.replace({'_': ' '}, regex=True);
final_df['GDP'] = pd.to_numeric(final_df['GDP'], errors='coerce');
final_df['Population'] = pd.to_numeric(final_df['Population'], errors='coerce');
#Only removes 2 countries to dropna
final_df = final_df.dropna(); 





def regression_vis(final_df):
    
    import plotly.express as px
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Assume final_df contains a column 'Country' which holds country names

    # Set features and target
    X = final_df[['GDP', 'Per_Capita_Waste_KG', 'Urban_population', 'Co2-Emissions']]
    y = final_df['Total_Plastic_Waste_MT']
    countries = final_df['Country']  # This will be used for hover information

    # Split and train
    X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(X, y, countries, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')



    plot_df = pd.DataFrame({
        'Actual Total Plastic Waste (MT)': y_test,
        'Predicted Total Plastic Waste (MT)': y_pred,
        'Country': countries_test
    })

    fig = px.scatter(plot_df, x='Actual Total Plastic Waste (MT)', y='Predicted Total Plastic Waste (MT)', 
                    hover_name='Country', 
                    title='Predicted vs Actual Total Plastic Waste')

    fig.add_shape(type='line', x0=plot_df['Actual Total Plastic Waste (MT)'].min(), x1=plot_df['Actual Total Plastic Waste (MT)'].max(),
                y0=plot_df['Actual Total Plastic Waste (MT)'].min(), y1=plot_df['Actual Total Plastic Waste (MT)'].max(),
                line=dict(color='red', dash='dash'))

    fig.update_layout(
        xaxis_title="Actual Total Plastic Waste (MT)",
        yaxis_title="Predicted Total Plastic Waste (MT)",
        showlegend=False
    )

    fig.show()


def top_plastic_vis(final_df):
    plt.figure(figsize=(12, 6))
    top_countries = final_df[['Country', 'Total_Plastic_Waste_MT']].sort_values(by='Total_Plastic_Waste_MT', ascending=False).head(10)
    sns.barplot(x='Total_Plastic_Waste_MT', y='Country', data=top_countries, palette='Blues_d')
    plt.title('Top 10 Countries by Total Plastic Waste (MT)')
    plt.xlabel('Total Plastic Waste (MT)')
    plt.ylabel('Country')
    plt.show()

def feature_heatmap(final_df):
    correlation_matrix = final_df[['Total_Plastic_Waste_MT', 'GDP','Urban_population', 'WaterPollution','Per_Capita_Waste_KG', 'AirQuality', 'Co2-Emissions', 'Recycling_Rate', 'Per_Capita_Waste_KG']].corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Key Features')
    plt.show()

def gdp_plastic_vis(final_df):
    fig = px.scatter(
    data_frame=final_df,
    x='GDP',
    y='Total_Plastic_Waste_MT',
    hover_name='Country',
    title='GDP vs. Total Plastic Waste',
    )

    fig.update_traces(marker=dict(line=dict(color='black', width=1)))

    fig.show()

def recycle_vis(final_df):
    plt.figure(figsize=(10, 6))
    corr_coeff = np.corrcoef(final_df['Recycling_Rate'], final_df['Total_Plastic_Waste_MT'])[0, 1]
    print(corr_coeff)
    plt.scatter(final_df['Recycling_Rate'], final_df['Total_Plastic_Waste_MT'], color='green', edgecolor='black', s=100)
    plt.title('Recycling Rate vs. Total Plastic Waste', fontsize=16, ha='center')
    plt.xlabel('Recycling Rate (%)', fontsize=12)
    plt.ylabel('Total Plastic Waste (Metric Tons)', fontsize=12)

    plt.show()
    
def industry_plastic_waste_vis(final_df):
    industry_counts = final_df['Main_Sources'].value_counts()
    plt.figure(figsize=(10, 6))
    industry_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Industries Contributing to Plastic Waste', fontsize=16)
    plt.xlabel('Leading Industry', fontsize=12)
    plt.ylabel('Number of Countries', fontsize=12)

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    
recycle_vis(final_df)

