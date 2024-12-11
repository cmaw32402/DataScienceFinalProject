import pandas as pd

plastic_waste_df = pd.read_csv(r'C:\Users\cmaw3\Desktop\DataScienceFinalProject\Plastic Waste Around the World.csv')
world_data_df = pd.read_csv(r'C:\Users\cmaw3\Desktop\DataScienceFinalProject\world-data-2023.csv')
pollution_df = pd.read_csv(r'C:\Users\cmaw3\Desktop\DataScienceFinalProject\cities_air_quality_water_pollution.18-10-2021.csv')

#sprint(world_data_df.columns)
#normalize world data
required_col_wd = ['Country', 'GDP', 'Population', 'Co2-Emissions', 'Urban_population']
world_data_df = world_data_df[required_col_wd]



#merge world data and plastic waste data
merged_df1 = pd.merge(plastic_waste_df, world_data_df, on='Country', how='inner')



pollution_df = pollution_df.replace('"', '', regex=True)

#normalize population data

required_col_pd = [' "Country"', ' "WaterPollution"', ' "AirQuality"']
pollution_df = pollution_df[required_col_pd]
pollution_df = pollution_df.groupby(' "Country"', as_index=False).mean()
pollution_df[' "Country"'] = pollution_df[' "Country"'].replace({
    ' United States of America': 'United States',
    " People's Republic of China": 'China',
    " Timor-Leste": "East Timor",
})
pollution_df[' "Country"'] = pollution_df[' "Country"'].str.strip()

pollution_df.rename(columns={' "Country"': 'Country', ' "AirQuality"': 'AirQuality', ' "WaterPollution"':'WaterPollution' }, inplace=True)
final_df = merged_df1.merge(pollution_df, on='Country', how='left')


final_df.to_csv('finaldata.csv', index=False)


















#print(pollution_df.columns)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000) 
#print(final_df)
#cut_off_countries = set(plastic_waste_df['Country']) - set(merged_df1['Country'])
#print(cut_off_countries)
#print(pollution_df2[' "WaterPollution"'])
#print(pollution_df2.columns)

#print(pollution_df['Country'])

#print(merged_df1['Country'])
#print(merged_df1)
#print(world_data_df.shape)
print(final_df)