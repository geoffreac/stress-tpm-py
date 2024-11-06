import numpy as np
import pandas as pd
import pickle
from scipy.linalg import expm  
import os
os.chdir('./stress-tpm-py/')
from datagen import load_or_generate_S_all, load_or_generate_dt, load_or_generate_full_data


# Function to calculate transition matrix nij between start_date and end_date
def count_nij(dt, S_all, start_date, end_date):
    order_levels = dt['R'].cat.categories
    matrix_nij = pd.DataFrame(0, index=order_levels, columns=order_levels)

    S_seq1 = S_all[(S_all['time'] == start_date) & (S_all['Rating'].notna())]
    group1 = S_seq1['Corporate'].unique()

    for corp in group1:
        rating1 = dt.loc[(dt['OBNAME'] == corp) & (dt['RAD'] <= start_date), 'R'].iloc[-1]
        list_rating = dt.loc[(dt['OBNAME'] == corp) & (dt['RAD'] > start_date) & (dt['RAD'] <= end_date), 'R']

        if "D" in list_rating.values:
            rating2 = "D"
        elif list_rating.empty:
            rating2 = rating1
        else:
            rating2 = list_rating.iloc[-1]

        matrix_nij.loc[rating1, rating2] += 1

    return matrix_nij

# Function to normalize nij matrix and calculate p transition probabilities
def p(nij_matrix):
    p_matrix = nij_matrix.div(nij_matrix.sum(axis=1), axis=0).fillna(0)
    
    # Special handling for "D" 
    p_matrix.loc['D', :] = 0
    p_matrix.loc['D', 'D'] = 1
    
    # Adjust probabilities for cohort method
    pc_matrix = p_matrix.copy()
    for i in pc_matrix.index:
        row_sum = pc_matrix.loc[i, :].sum()
        pc_matrix.loc[i, :] += pc_matrix.loc[i, :] * (1 - row_sum) / row_sum if row_sum != 0 else 0

    pc_matrix.loc['D', :] = 0
    pc_matrix.loc['D', 'D'] = 1
    if 'NR' in pc_matrix.index:
        pc_matrix = pc_matrix.drop('NR', axis=0).drop('NR', axis=1)
    
    return pc_matrix

# Function to calculate cohort transitions
def cohort(dt, S_all, seqtime):
    column_names = dt['R'].unique()
    row_names = column_names
    seqnumber = len(seqtime) - 1

    nij = np.zeros((len(row_names), len(column_names), seqnumber))

    for k in range(seqnumber):
        nij[:, :, k] = count_nij(dt, S_all, seqtime[k], seqtime[k+1]).values

    p_matrices = np.zeros_like(nij)
    
    # Calculate transition probabilities
    for k in range(seqnumber):
        nij_k = nij[:, :, k]
        row_sums = nij_k.sum(axis=1, keepdims=True)
        p_matrices[:, :, k] = np.divide(nij_k, row_sums, where=row_sums != 0)
        p_matrices[7, :, k] = 0  # Handle default state ("D")
        p_matrices[7, 7, k] = 1

    # Cohort adjustment (handling "NR" as no information)
    pc_matrices = np.copy(p_matrices)
    for k in range(seqnumber):
        for i in range(len(row_names)):
            row_sum = pc_matrices[i, :, k].sum()
            if row_sum != 0:
                pc_matrices[i, :, k] += pc_matrices[i, :, k] * (1 - row_sum) / row_sum
        pc_matrices[7, :, k] = 0  # Default state remains unchanged
        pc_matrices[7, 7, k] = 1
    
    return {'nij': nij, 'p': p_matrices, 'pc': pc_matrices}

# Function to compute Nij transitions for duration calculations
def count_Nij(S_all, start_date, end_date):
    S = S_all[(S_all['time'] == start_date) & S_all['Rating'].notna()]
    group = S['Corporate'].unique()
    row_names = S_all['R'].unique()
    column_names = row_names

    Nij = pd.DataFrame(np.zeros((len(row_names), len(column_names))), index=row_names, columns=column_names)

    temp = S_all[(S_all['time'] >= start_date) & (S_all['time'] <= end_date) & S_all['Corporate'].isin(group)]
    
    for corp in group:
        temp_corp = temp[temp['Corporate'] == corp]
        if len(temp_corp['Rating'].unique()) > 1:
            for j in range(len(temp_corp) - 1):
                rating1 = temp_corp['Rating'].iloc[j]
                rating2 = temp_corp['Rating'].iloc[j+1]
                if rating1 != rating2:
                    Nij.loc[rating1, rating2] += 1
    return Nij

# Function to compute integral over the time period for duration calculations
def Int_f(S_all, start_date, end_date):
    S = S_all[(S_all['time'] == start_date) & S_all['Rating'].notna()]
    group = S['Corporate'].unique()

    row_names = S_all['R'].unique()
    matrix_int_Y = pd.DataFrame(0, index=row_names, columns=[0])

    for i in row_names:
        integral_Y = S_all[(S_all['Corporate'].isin(group)) & (S_all['Rating'] == i) & 
                           (S_all['time'] > start_date) & (S_all['time'] <= end_date)]
        integral_Y = integral_Y.groupby('time').size().sum() / (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        matrix_int_Y.loc[i, 0] = integral_Y

    return matrix_int_Y

# Function to compute lambda_ij transition intensities for duration
def lambdaij_f(S_all, Nij_t, Int_t, start_date, end_date):
    row_names = S_all['R'].unique()
    column_names = row_names

    lambdaij = pd.DataFrame(np.zeros((len(row_names), len(column_names))), index=row_names, columns=column_names)

    for i in row_names:
        for j in column_names:
            if i != j and Int_t.loc[i, 0] != 0:
                lambdaij.loc[i, j] = Nij_t.loc[i, j] / Int_t.loc[i, 0]

    lambdaij.loc['D', :] = 0

    for i in row_names:
        lambdaij.loc[i, i] = -lambdaij.loc[i, lambdaij.columns != i].sum()

    return lambdaij

# Function to compute the transition matrix based on lambda_ij (duration method)
def duration_f(lambdaij):
    return expm(lambdaij.values)

# Example usage of the functions:
start_date = '2011-03-01'
end_date = '2019-03-01'
seqtime_quarter = pd.date_range(start=start_date, end=end_date, freq='Q')

xml_directory = './data_source/SP-Corporate-2020-11-01'
csv_output_path = './data_source/generated_data/full_data.csv'
full_data = load_or_generate_full_data(xml_directory, csv_output_path)

dt_path ='./data_source/generated_data/dt.pkl'
dt = load_or_generate_dt(full_data,dt_path)

S_all_path = './data_source/generated_data/S_all.pkl'
S_all = load_or_generate_S_all(dt, S_all_path)

# Define start and end dates
start_date = '2011-03-01'
end_date = '2019-03-01'

# Create a sequence of quarters between start and end dates
seqtime_quarter = pd.date_range(start=start_date, end=end_date, freq='Q')
T_end = len(seqtime_quarter)

# Create arrays (lists of lists) to store the results for cohort and duration matrices
p_cohort = [[None for _ in range(T_end - 1)] for _ in range(T_end - 1)]
p_duration = [[None for _ in range(T_end - 1)] for _ in range(T_end - 1)]
nij_all = [[None for _ in range(T_end - 1)] for _ in range(T_end - 1)]
Nij = [[None for _ in range(T_end - 1)] for _ in range(T_end - 1)]
Int = [[None for _ in range(T_end - 1)] for _ in range(T_end - 1)]
lambdaij = [[None for _ in range(T_end - 1)] for _ in range(T_end - 1)]

# Loop through each combination of time intervals
T_max = range(1, T_end)

# for i in T_max:
#     print(f"i = {i}")
i = 4 # T= 1 year
for j in range(T_end - i - 1):
    print(f"j = {j}")

    # Cohort calculations
    nij_matrix = count_nij(dt, S_all, seqtime_quarter[j], seqtime_quarter[j+i])
    nij_all[i][j] = nij_matrix
    p_cohort[i][j] = p(nij_matrix)

    # Duration calculations
    # Nij_matrix = count_Nij(S_all, seqtime_quarter[j], seqtime_quarter[j+i])
    # Nij[i][j] = Nij_matrix

    # Int_matrix = Int_f(S_all, seqtime_quarter[j], seqtime_quarter[j+i])
    # Int[i][j] = Int_matrix

    # lambdaij_matrix = lambdaij_f(S_all, Nij_matrix, Int_matrix, seqtime_quarter[j], seqtime_quarter[j+i])
    # lambdaij[i][j] = lambdaij_matrix

    # p_duration[i][j] = duration_f(lambdaij_matrix)

# Save the results to files
with open('./data_source/generated_data/p_cohort.pkl', 'wb') as f:
    pickle.dump(p_cohort, f)

with open('./data_source/generated_data/nij_all.pkl', 'wb') as f:
    pickle.dump(nij_all, f)

# with open('./data_source/generated_data/p_duration.pkl', 'wb') as f:
#     pickle.dump(p_duration, f)

# with open('./data_source/generated_data/Int.pkl', 'wb') as f:
#     pickle.dump(Int, f)

# with open('./data_source/generated_data/Nij.pkl', 'wb') as f:
#     pickle.dump(Nij, f)

# with open('./data_source/generated_data/lambdaij.pkl', 'wb') as f:
#     pickle.dump(lambdaij, f)