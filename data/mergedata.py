import pandas as pd

binaryreference = '../binary_oxide_reference_data/data.txt'
generateddata = 'generated_features.xlsx'

x = pd.read_csv(binaryreference)
y = pd.read_excel(generateddata)

y.rename(columns={'Simulated Composition': 'composition'}, inplace=True)

y['composition'] = y['composition'].replace(
                                            '\n',
                                            '',
                                            regex=True
                                            )


df = pd.merge(x, y, on=['composition'])

df.to_excel('merged.xlsx', index=False)
