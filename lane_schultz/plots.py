from matplotlib import pyplot as pl

import pandas as pd
import os

filename = 'stats.txt'
paths = os.walk('./')

alphadata = []
for path in paths:
    alphacondition = filename in path[2]
    alphacondition = alphacondition and 'split' not in path[0]
    alphacondition = alphacondition and 'alpha' in path[0]

    if alphacondition:
        filepath = os.path.join(path[0], filename)

        # Gathers the stat data for the last set of values (TEST)
        data = {}
        with open(filepath) as item:
            for line in item:
                values = line.strip().split(' ')

                if len(values) > 1:
                    try:
                        for i in values[1:]:
                            data[values[0][:-1]] = float(values[1])
                    except Exception:
                        pass

            alpha = path[0].split('/')[1]
            alpha = alpha.split('_')[1]
            if 'p' in alpha:
                alpha = alpha.replace('p', '.')
                       
            alpha = float(alpha) 
            data['alpha'] = alpha

        alphadata.append(data)


alphadf = pd.DataFrame(alphadata)

columns = [i for i in alphadf.columns if i != 'alpha']
for col in columns:

    alphafig, alphaax = pl.subplots()

    alphaax.plot(
                 alphadf['alpha'],
                 alphadf[col],
                 marker='.',
                 linestyle='none'
                 )

    alphaax.set_xlabel('alpha')
    alphaax.set_ylabel(col)
    alphaax.grid()

alphafig.tight_layout()
pl.show()
print(alphadf)
