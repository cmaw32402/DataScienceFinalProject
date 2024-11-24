import pandas as pd

plastic_waste_df = pd.read_csv(r'C:\Users\cmaw3\Desktop\DataScienceFinalProject\Plastic Waste Around the World.csv')
world_data_df = pd.read_csv(r'C:\Users\cmaw3\Desktop\DataScienceFinalProject\world-data-2023.csv')
pollution_df = pd.read_csv(r'C:\Users\cmaw3\Desktop\DataScienceFinalProject\cities_air_quality_water_pollution.18-10-2021.csv')
#print(d

required_col_1 = ['Country', 'GDP']
world_data_df2 = world_data_df[required_col_1]

required_col_2 = ['Country', 'GDP']
merged_df1 = pd.merge(plastic_waste_df, world_data_df2, on='Country', how='inner')

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000) 
#cut_off_countries = set(plastic_waste_df['Country']) - set(merged_df1['Country'])
#print(cut_off_countries)
print(world_data_df['Country'])

#print(merged_df1)
#print(world_data_df.shape)
