# these functions are for the notebook for exploring ML training with coupon validation
import numpy as np
import pandas as pd

def find_closest_valid(row: pd.Series, df: pd.DataFrame) -> [int, int]:
    # Filter the DataFrame to get the same prefix and mastercode and valid rows
    filtered = df[(df['prefix'] == row['prefix']) &
                  (df['mastercode'] == row['mastercode']) &
                  (df['valid'] == True)]

    # Rows with random_digits less than the current row (behind)
    behind = filtered[filtered['random_digits'] < row['random_digits']]

    # Rows with random_digits greater than the current row (in front)
    front = filtered[filtered['random_digits'] > row['random_digits']]

    # Find the closest valid row behind (if it exists)
    if not behind.empty:
        behind['distance'] = np.abs(behind['random_digits'] - row['random_digits'])
        closest_behind = behind.loc[behind['distance'].idxmin()]['distance']
    else:
        closest_behind = None  # No valid row behind

    # Find the closest valid row in front (if it exists)
    if not front.empty:
        front['distance'] = np.abs(front['random_digits'] - row['random_digits'])
        closest_front = front.loc[front['distance'].idxmin()]['distance']

    else:
        closest_front = None  # No valid row in front

    return closest_behind, closest_front


def find_closest_valid_optimized(df):

    # Ensure the DataFrame is sorted by 'prefix', 'mastercode', and 'random_digits'
    df = df.sort_values(by=['prefix', 'mastercode', 'random_digits']).copy()

    # Group by 'prefix' and 'mastercode'
    grouped = df.groupby(['prefix', 'mastercode'])

    # For each group, shift the valid 'random_digits' for forward and backward distances
    df['prev_random_digits'] = grouped['random_digits'].shift(1)  # Previous valid row
    df['next_random_digits'] = grouped['random_digits'].shift(-1)  # Next valid row

    # Calculate distances only for valid rows
    df['closest_behind'] = np.where(df['valid'], df['random_digits'] - df['prev_random_digits'], np.nan)
    df['closest_front'] = np.where(df['valid'], df['next_random_digits'] - df['random_digits'], np.nan)

    return df