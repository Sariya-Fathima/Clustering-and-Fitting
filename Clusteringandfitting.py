
# In[211]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# # Dataframe 1 : API.CSV

# In[212]:
def read_csv_with_skiprows(file_path, skip_rows=0):
    """
    Read a CSV file into a DataFrame using pandas, skipping specified number of rows.

    Parameters:
    - file_path (str): The path to the CSV file.
    - skip_rows (int): Number of rows to skip from the beginning of the file (default is 0).

    Returns:
    - pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    # Read CSV file with specified skiprows parameter
    df = pd.read_csv(file_path, skiprows=skip_rows)

    return df

# Example usage:
file_path = "dataset/api.csv"
skip_rows = 3
df1 = read_csv_with_skiprows(file_path, skip_rows)

# Display DataFrame
print(df1.head())


# In[213]:


total_rows = len(df1)
total_rows


# In[214]:


# # Drop the 'Unnamed: 67' column as it appears to be empty
# df1 = df.drop(columns=['Unnamed: 67'])
# Convert columns to appropriate data types
numeric_columns = df1.columns[4:]
df1[numeric_columns] = df1[numeric_columns].apply(pd.to_numeric, errors='coerce')
# Drop rows with all NaN values
df1 = df1.dropna(how='all')
# Forward-fill missing values in the 'Country Name' and 'Country Code' columns
df1['Country Name'].fillna(method='ffill', inplace=True)
df1['Country Code'].fillna(method='ffill', inplace=True)
# Display the cleaned DataFrame
print("\nCleaned DataFrame:\n")
df1.head()


# In[215]:



# Filter rows where "Indicator Name" is "Urban population growth (annual %)"
selected_rows = df1[df1['Indicator Name'] == 'Urban population growth (annual %)']

# Extract values from columns "1960" and "1961"
values_1960 = selected_rows['1960'].tolist()
values_1977 = selected_rows['1977'].tolist()

print("Values from 1960 column:", values_1960)
print("Values from 1977 column:", values_1977)


# In[216]:




def exponential_growth(x, a, b, c):
    return a * np.exp(b * (x - x.iloc[0])) + c

def fit_exponential_growth(df, column_name):
    # Drop rows with NaN values in the specified column
    df = df.dropna(subset=[column_name])

    # Use curve_fit to fit the data to the model
    params, covariance = curve_fit(exponential_growth, df['Year'], df[column_name])

    # Extract the optimized parameters
    a_opt, b_opt, c_opt = params

    # Generate fitted curve using the optimized parameters
    fitted_curve = exponential_growth(df['Year'], a_opt, b_opt, c_opt)

    # Plot the original data and the fitted curve
    plt.scatter(df['Year'], df[column_name], label='Original Data')
    plt.plot(df['Year'], fitted_curve, label='Fitted Curve', color='red')
    plt.title(f'Exponential Growth Fitting for {column_name}')
    plt.xlabel('Year')
    plt.ylabel(f'{column_name}')
    plt.legend()
    plt.show()

# Example usage with your DataFrame
data = {
    'Year': np.arange(1960, 1977),
    'Urban population growth (annual %)': [np.nan, 2.14785808691749, 1.52032913978224, 1.35704159491957, 1.18647193196888, 1.00157648441363, 0.835370915364582,
                                          0.358732863284107, -0.119434690065009, -0.26592224153399, -0.406897803172485, -0.522719324507363, 0.0369480883567942, 0.832738955957845,
                                          1.07987402025453, 1.10744554389865, 0.753103604875798]
}

df = pd.DataFrame(data)

# Get unique column names excluding 'Year'
columns_to_fit = [col for col in df.columns if col != 'Year']

# Fit exponential growth model for each column
for column_name in columns_to_fit:
    fit_exponential_growth(df, column_name)


# In[217]:


# Display basic statistics
print("\nBasic Statistics:\n")
df1.describe()


# In[218]:


# Display trends over the years for a specific country (e.g., Aruba)
aruba_data = df1[df1['Country Name'] == 'Aruba']
print("\nTrends for Aruba:\n")
aruba_data

# Display trends for a specific indicator (e.g., Urban population (% of total population))
urban_population_percentage = df1[df1['Indicator Code'] == 'SP.URB.TOTL.IN.ZS']
print("\nTrends for Urban population (% of total population):\n")
print(urban_population_percentage)


# Set the style for seaborn
sns.set(style="whitegrid")

# Visualize the trends for a specific indicator (e.g., Urban population (% of total population))
plt.figure(figsize=(16, 8))
sns.barplot(data=urban_population_percentage, x='Country Code', y='1960', label='1960', color='blue')
sns.barplot(data=urban_population_percentage, x='Country Code', y='1970', label='1970', color='orange')
sns.barplot(data=urban_population_percentage, x='Country Code', y='1980', label='1980', color='green')
sns.barplot(data=urban_population_percentage, x='Country Code', y='1990', label='1990', color='red')
sns.barplot(data=urban_population_percentage, x='Country Code', y='2000', label='2000', color='purple')
sns.barplot(data=urban_population_percentage, x='Country Code', y='2010', label='2010', color='brown')

# Set font size for the text on the graph, x-axis, and y-axis labels
plt.title('Urban Population Trends Over the Decades (% of Total Population)', fontsize=18)
plt.xlabel('Country Code', fontsize=8)
plt.ylabel('Urban Population (% of Total Population)', fontsize=14)
plt.legend()
plt.show()


# Set the seaborn context with a smaller font size
sns.set_context("notebook", rc={"font.size": 10})

# Melt the DataFrame to reshape it for visualization
melted_aruba_data = pd.melt(aruba_data, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                            var_name='Year', value_name='Population')

# Convert 'Year' to numeric (remove 'YR' prefix)
melted_aruba_data['Year'] = melted_aruba_data['Year'].str.extract('(\d+)').astype(float)

# Visualize the trends for Aruba using a scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=melted_aruba_data, x='Year', y='Population', hue='Indicator Name', s=50)
plt.title('Population Trends for Aruba Over the Decades', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Population', fontsize=8)
plt.legend(title='Indicator', fontsize=8)
plt.show()

# Create a new column 'Decade' based on 'Year'
melted_aruba_data['Decade'] = (melted_aruba_data['Year'] // 10) * 10

# Group by 'Decade' and calculate the sum of 'Population' for each decade
population_by_decade = melted_aruba_data.groupby('Decade')['Population'].sum().reset_index()

# Set the seaborn context with a smaller font size
sns.set_context("notebook", rc={"font.size": 10})

# Visualize the population counts by decades using a bar graph
plt.figure(figsize=(12, 6))
sns.barplot(data=population_by_decade, x='Decade', y='Population', color='skyblue')
plt.title('Population Counts for Aruba by Decades', fontsize=14)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Population Count', fontsize=8)
plt.show()


# In[219]:


# Display trends over the years for a specific country (e.g., Aruba)
aruba_data = df1[df1['Country Name'] == 'Africa Eastern and Southern']
print("\nAfrica Eastern and Southern:\n")
aruba_data

# Display trends for a specific indicator (e.g., Urban population (% of total population))
urban_population_percentage = df1[df1['Indicator Code'] == 'SP.URB.TOTL.IN.ZS']
print("\nTrends for Urban population (% of total population):\n")
print(urban_population_percentage)


# Set the style for seaborn
sns.set(style="whitegrid")

# Visualize the trends for a specific indicator (e.g., Urban population (% of total population))
plt.figure(figsize=(16, 8))
sns.barplot(data=urban_population_percentage, x='Country Code', y='1960', label='1960', color='blue')
sns.barplot(data=urban_population_percentage, x='Country Code', y='1970', label='1970', color='orange')
sns.barplot(data=urban_population_percentage, x='Country Code', y='1980', label='1980', color='green')
sns.barplot(data=urban_population_percentage, x='Country Code', y='1990', label='1990', color='red')
sns.barplot(data=urban_population_percentage, x='Country Code', y='2000', label='2000', color='purple')
sns.barplot(data=urban_population_percentage, x='Country Code', y='2010', label='2010', color='brown')

# Set font size for the text on the graph, x-axis, and y-axis labels
plt.title('Urban Population Trends Over the Decades (% of Total Population)', fontsize=18)
plt.xlabel('Country Code', fontsize=8)
plt.ylabel('Urban Population (% of Total Population)', fontsize=14)
plt.legend()
plt.show()


# Set the seaborn context with a smaller font size
sns.set_context("notebook", rc={"font.size": 10})

# Melt the DataFrame to reshape it for visualization
melted_aruba_data = pd.melt(aruba_data, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                            var_name='Year', value_name='Population')

# Convert 'Year' to numeric (remove 'YR' prefix)
melted_aruba_data['Year'] = melted_aruba_data['Year'].str.extract('(\d+)').astype(float)

# Visualize the trends for Aruba using a scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=melted_aruba_data, x='Year', y='Population', hue='Indicator Name', s=50)
plt.title('Population Trends for Aruba Over the Decades', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Population', fontsize=8)
plt.legend(title='Indicator', fontsize=8)
plt.show()

# Create a new column 'Decade' based on 'Year'
melted_aruba_data['Decade'] = (melted_aruba_data['Year'] // 10) * 10

# Group by 'Decade' and calculate the sum of 'Population' for each decade
population_by_decade = melted_aruba_data.groupby('Decade')['Population'].sum().reset_index()

# Set the seaborn context with a smaller font size
sns.set_context("notebook", rc={"font.size": 10})

# Visualize the population counts by decades using a bar graph
plt.figure(figsize=(12, 6))
sns.barplot(data=population_by_decade, x='Decade', y='Population', color='skyblue')
plt.title('Population Counts for Aruba by Decades', fontsize=14)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Population Count', fontsize=8)
plt.show()


# In[220]:


# Display trends over the years for a specific country (e.g., Aruba)
aruba_data = df1[df1['Country Name'] == 'Afghanistan']
print("\nAfghanistan:\n")
aruba_data

# Display trends for a specific indicator (e.g., Urban population (% of total population))
urban_population_percentage = df1[df1['Indicator Code'] == 'SP.URB.TOTL.IN.ZS']
print("\nTrends for Urban population (% of total population):\n")
print(urban_population_percentage)


# Set the style for seaborn
sns.set(style="whitegrid")

# Visualize the trends for a specific indicator (e.g., Urban population (% of total population))
plt.figure(figsize=(16, 8))
sns.barplot(data=urban_population_percentage, x='Country Code', y='1960', label='1960', color='blue')
sns.barplot(data=urban_population_percentage, x='Country Code', y='1970', label='1970', color='orange')
sns.barplot(data=urban_population_percentage, x='Country Code', y='1980', label='1980', color='green')
sns.barplot(data=urban_population_percentage, x='Country Code', y='1990', label='1990', color='red')
sns.barplot(data=urban_population_percentage, x='Country Code', y='2000', label='2000', color='purple')
sns.barplot(data=urban_population_percentage, x='Country Code', y='2010', label='2010', color='brown')

# Set font size for the text on the graph, x-axis, and y-axis labels
plt.title('Urban Population Trends Over the Decades (% of Total Population)', fontsize=18)
plt.xlabel('Country Code', fontsize=8)
plt.ylabel('Urban Population (% of Total Population)', fontsize=14)
plt.legend()
plt.show()


# Set the seaborn context with a smaller font size
sns.set_context("notebook", rc={"font.size": 10})

# Melt the DataFrame to reshape it for visualization
melted_aruba_data = pd.melt(aruba_data, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                            var_name='Year', value_name='Population')

# Convert 'Year' to numeric (remove 'YR' prefix)
melted_aruba_data['Year'] = melted_aruba_data['Year'].str.extract('(\d+)').astype(float)

# Visualize the trends for Aruba using a scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=melted_aruba_data, x='Year', y='Population', hue='Indicator Name', s=50)
plt.title('Population Trends for Aruba Over the Decades', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Population', fontsize=8)
plt.legend(title='Indicator', fontsize=8)
plt.show()

# Create a new column 'Decade' based on 'Year'
melted_aruba_data['Decade'] = (melted_aruba_data['Year'] // 10) * 10

# Group by 'Decade' and calculate the sum of 'Population' for each decade
population_by_decade = melted_aruba_data.groupby('Decade')['Population'].sum().reset_index()

# Set the seaborn context with a smaller font size
sns.set_context("notebook", rc={"font.size": 10})

# Visualize the population counts by decades using a bar graph
plt.figure(figsize=(12, 6))
sns.barplot(data=population_by_decade, x='Decade', y='Population', color='skyblue')
plt.title('Population Counts for Aruba by Decades', fontsize=14)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Population Count', fontsize=8)
plt.show()


# In[221]:


# Read the data into a DataFrame
df2 = pd.read_csv("dataset/metadata_api.csv")
df2.head(3)


# In[222]:


# Drop the 'Unnamed: 4' column
df2.drop('Unnamed: 4', axis=1, inplace=True)

# Perform analysis and visualization (example: count of records per organization)
organization_counts = df2['SOURCE_ORGANIZATION'].value_counts()

# Display analysis result
print("Count of records per organization:")
print(organization_counts)


# In[223]:


# Visualization (bar chart)
plt.figure(figsize=(10, 6))
organization_counts.plot(kind='bar', color='skyblue')
plt.title('Count of Records per Organization')
plt.xlabel('Organization')
plt.ylabel('Count')
plt.show()


# In[224]:


# Analysis 1: Count of records per organization
organization_counts = df2['SOURCE_ORGANIZATION'].value_counts()
print("Count of records per organization:")
print(organization_counts)


# In[225]:


# Visualization 1 (bar chart)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
organization_counts.plot(kind='bar', color='skyblue')
plt.title('Count of Records per Organization')
plt.xlabel('Organization')
plt.ylabel('Count')

# Analysis 2: Average length of SOURCE_NOTE for each INDICATOR_NAME
df2['SOURCE_NOTE_LENGTH'] = df2['SOURCE_NOTE'].str.len()
average_note_length = df2.groupby('INDICATOR_NAME')['SOURCE_NOTE_LENGTH'].mean()
print("\nAverage length of SOURCE_NOTE for each INDICATOR_NAME:")
print(average_note_length)

# Visualization 2 (bar chart)
plt.subplot(1, 2, 2)
average_note_length.plot(kind='bar', color='lightcoral')
plt.title('Average Length of SOURCE_NOTE for Each Indicator')
plt.xlabel('Indicator Name')
plt.ylabel('Average Length')
plt.tight_layout()
plt.show()


# ## Dataset 3 : MetaData_country.csv

# In[226]:


# Read the data into a DataFrame
df3 = pd.read_csv("dataset/metadata_country.csv")
df3.head(3)


# In[227]:


# Drop the 'Unnamed: 4' column
df3.drop('Unnamed: 5', axis=1, inplace=True)
df3.head()


# In[228]:


# Analysis 1: Count of countries in each region
region_counts = df3['Region'].value_counts()
print("Count of countries in each region:")
print(region_counts)


# In[229]:


# Visualization 1 (bar chart)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
region_counts.plot(kind='bar', color='skyblue')
plt.title('Count of Countries in Each Region')
plt.xlabel('Region')
plt.ylabel('Count')


# In[230]:


# Analysis 2: Distribution of countries across income groups
income_distribution = df3['IncomeGroup'].value_counts()
print("\nDistribution of countries across income groups:")
print(income_distribution)


# In[231]:


# Visualization 2 (pie chart)
plt.subplot(1, 2, 2)
income_distribution.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=90)
plt.title('Distribution of Countries Across Income Groups')
plt.ylabel('')  # Remove the default 'ylabel'
plt.tight_layout()
plt.show()


# In[232]:


# Create a pivot table
pivot_table = pd.pivot_table(df3, values='Country Code', index='Region', columns='IncomeGroup', aggfunc='count')

# Visualization (heatmap)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt='g', cbar_kws={'label': 'Count of Countries'})
plt.title('Pivot Chart: Count of Countries by Region and Income Group')
plt.show()


# In[ ]:




