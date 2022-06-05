import pandas as pd
import numpy as np
import logging

for year in range(2018, 2019):
    for race in range(1, 21):
        try:
            data = pd.read_csv(f"races/{year}/{race}/overall.csv", low_memory=False)
            print(f'race {year}, {race} started')
            columns = data.columns.tolist()
            if 'Unnamed: 0' in columns:
                columns.remove('Unnamed: 0')
            df = pd.DataFrame(columns = columns)

            for _, row in data.iterrows():
                if np.isnan(row['Overtake']):
                    row_fixed = np.array(row.values[:-1])
                else:
                    row_fixed = np.array(row.values[1:])
                data_to_append = {}
                for i in range(len(columns)):
                    data_to_append[columns[i]] = [row_fixed[i]]

                df = pd.concat([df, pd.DataFrame.from_dict(data_to_append)], ignore_index=True)

            df.to_csv(f"races/fixed/df_{year}_{race}.csv")
            print(f'race {year}, {race} completed')
        except FileNotFoundError:
            logging.error(f'race {year}, {race} not found')