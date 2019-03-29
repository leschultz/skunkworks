from pymatgen.analysis.reaction_calculator import ComputedReaction

import pymatgen as mg
import pandas as pd

import re

mpr = mg.MPRester()  # My API key for pymatgen

epath = '../data/vaspdata_collected.xlsx'

df = pd.read_excel(epath)  # Import data containing composition energy
columns = list(df.columns)[:-2]  # Relevant columns

df = df[columns]  # Compositions and energies
df[columns[0]] = df[columns[0]].replace('\n', '', regex=True)

print(df)

for row in df['composition']:
    elements = re.findall('[A-Z][a-z]?|[0-9]+', row)
    elements = [i for i in elements if not i.isdigit()]

    oxides = [str(i)+'-O' for i in elements if str(i) != 'O']

    oxidechoices = []
    for oxide in oxides:
        d = mpr.query(oxide, ['pretty_formula', 'e_above_hull', 'energy_per_atom'])

        d = pd.DataFrame(d)
        d = d.sort_values(by=['e_above_hull', 'energy_per_atom'])
        d = d.reset_index(drop=True)

        choice = d['pretty_formula'][0]

        oxidechoices.append(choice)


    # Oxide energies
    print(row)
    print(oxidechoices)
    reactants = [mpr.query(i, ['energy']) for i in oxidechoices]

    print(reactants)
    # Calculate reaction energies
