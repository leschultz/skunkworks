from pymatgen.analysis.reaction_calculator import ComputedReaction

import pymatgen as mg
import pandas as pd
import numpy as np

import re


def natoms(x):
    '''
    Sum the nubmer of atoms in a composition.
    '''

    return sum([j for i, j in x.items()])

mpr = mg.MPRester()  # My API key for pymatgen

epath = '../data/vaspdata_collected.xlsx'

df = pd.read_excel(epath)  # Import data containing composition energy

df = df[['composition', 'energy']]
df['composition'] = df['composition'].replace(' \n', '', regex=True)

# Find the number of atoms in each composition
atoms = df['composition'].apply(mg.Composition)
atoms = atoms.apply(dict)
atoms = atoms.apply(natoms)

df['atoms'] = atoms
df['e_per_atom'] = df['energy']/df['atoms']

df['oxides'] = np.nan
df['oxide_e_per_atom'] = np.nan
df['e_per_atom-sum(oxide_e_per_atom)'] = np.nan

count = 0
for row, e in zip(df['composition'], df['e_per_atom']):
    elements = re.findall('[A-Z][a-z]?|[0-9]+', row)
    elements = [i for i in elements if not i.isdigit()]

    oxides = [str(i)+'-O' for i in elements if str(i) != 'O']

    oxidechoices = []
    energies = []
    for oxide in oxides:
        d = mpr.query(oxide, ['pretty_formula', 'e_above_hull', 'energy_per_atom'])

        d = pd.DataFrame(d)
        d = d.sort_values(by=['e_above_hull', 'energy_per_atom'])
        d = d.reset_index(drop=True)

        oxidechoices.append(d['pretty_formula'][0])
        energies.append(d['energy_per_atom'][0])

    energysum = sum(energies)

    diff = e-energysum

    df['oxides'].iloc[count] = oxides
    df['oxide_e_per_atom'].iloc[count] = energysum
    df['e_per_atom-sum(oxide_e_per_atom)'].iloc[count] = diff

    count += 1

df.to_csv('data.txt', index=False)
