import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from exploit import inertia, bias, DR

def stretch(matrix, alpha, beta, correct=True):
    if correct:
        matrix = matrix.copy()
        for i in range(1, matrix.shape[0]):
            if np.sum(matrix[i, :i]) == 0:
                matrix[i, i-1] = 0.0001
    
    # Step a
    matrix1 = matrix.copy()
    diag_indices = np.diag_indices_from(matrix1)
    matrix1[diag_indices] *= (1 - alpha)
    
    for i in range(matrix1.shape[0] - 1):
        for j in range(matrix1.shape[1]):
            if i != j:
                matrix1[i, j] *= (1 + alpha * matrix1[i, i] / np.sum(matrix[i, :i]))
    
    # Step b
    matrix2 = matrix1.copy()
    if beta >= 0:
        upper_indices = np.triu_indices_from(matrix1, k=1)
        matrix2[upper_indices] *= (1 - beta)
        
        for i in range(matrix2.shape[0]):
            for j in range(matrix2.shape[1]):
                if i > j:
                    ki = np.sum(matrix1[i, i+1:]) / np.sum(matrix1[i, :i])
                    matrix2[i, j] *= (1 + beta * ki)
    else:
        lower_indices = np.tril_indices_from(matrix1, k=-1)
        matrix2[lower_indices] *= (1 + beta)
        
        for i in range(matrix2.shape[0]):
            for j in range(matrix2.shape[1]):
                if i < j:
                    li = np.sum(matrix1[i, :i]) / np.sum(matrix1[i, i+1:])
                    matrix2[i, j] *= (1 - beta * li)
    
    matrix2[0, :] /= np.sum(matrix2[0, :])
    return matrix2

def error(para, base_matrix, target_matrix):
    alpha, beta = para
    stretched_matrix = stretch(base_matrix, alpha, beta)
    return (
        10 * (DR(target_matrix) - DR(stretched_matrix))**2 +
        (bias(target_matrix) - bias(stretched_matrix))**2 +
        (inertia(target_matrix) - inertia(stretched_matrix))**2
    )

def construct(base_matrix, target_matrix):
    results = []
    if isinstance(target_matrix, np.ndarray) and target_matrix.ndim == 3:
        for i in range(target_matrix.shape[2]):
            result = minimize(error, [0.01, 0.01], args=(base_matrix, target_matrix[:, :, i]))
            results.append((result.x[0], result.x[1]))
    elif isinstance(target_matrix, list):
        for matrix in target_matrix:
            result = minimize(error, [0.01, 0.01], args=(base_matrix, matrix))
            results.append((result.x[0], result.x[1]))
    else:
        result = minimize(error, [0.01, 0.01], args=(base_matrix, target_matrix))
        results.append((result.x[0], result.x[1]))
    
    # Reconstruct TPMs based on obtained parameters
    reconstructed_matrices = []
    biases, inertias, DRs = [], [], []
    
    for alpha, beta in results:
        stretched_matrix = stretch(base_matrix, alpha, beta)
        reconstructed_matrices.append(stretched_matrix)
        biases.append(bias(stretched_matrix))
        inertias.append(inertia(stretched_matrix))
        DRs.append(DR(stretched_matrix))
    
    return {
        'matrix': reconstructed_matrices,
        'bias': biases,
        'inertia': inertias,
        'DR': DRs,
        'params': results
    }

def generate_data_tpm_reconstructed(T_end, seqtime_quarter, list_metrics, p_cohort):
    datalist = []
    
    for i in range(1, T_end):
        for j in range(T_end - i):
            start_date = seqtime_quarter[j]
            dat = pd.DataFrame({'Time': seqtime_quarter[j:T_end - i]})
            dat['Inertia'] = list_metrics[i, j]['inertia_cohort']
            dat['Bias'] = list_metrics[i, j]['bias_cohort']
            dat['DR'] = list_metrics[i, j]['DR_cohort']
            dat['Method'] = "Cohort"
            dat['Delta'] = i
            dat['Start'] = start_date
            datalist.append(dat)
    
    for i in range(1, T_end):
        for j in range(T_end - i):
            start_date = seqtime_quarter[j]
            dat = pd.DataFrame({'Time': seqtime_quarter[j:T_end - i]})
            dat['Inertia'] = list_metrics[i, j]['inertia_duration']
            dat['Bias'] = list_metrics[i, j]['bias_duration']
            dat['DR'] = list_metrics[i, j]['DR_duration']
            dat['Method'] = "Duration"
            dat['Delta'] = i
            dat['Start'] = start_date
            datalist.append(dat)
    
    for i in range(1, 10):
        for j in range(min(T_end - i, 4)):
            construct_cohort = construct(list_metrics[i, j]['avg_cohort'], p_cohort[i, j:T_end - i])
            dat = pd.DataFrame({'Time': seqtime_quarter[j:T_end - i]})
            dat['Inertia'] = construct_cohort['inertia']
            dat['Bias'] = construct_cohort['bias']
            dat['DR'] = construct_cohort['DR']
            dat['Method'] = "Cons Avg Cohort"
            dat['Delta'] = i
            dat['Start'] = seqtime_quarter[j]
            datalist.append(dat)
    
    return pd.concat(datalist, ignore_index=True)

# Example usage
data_tpm_reconstructed = generate_data_tpm_reconstructed(T_end, seqtime_quarter, list_metrics)


def plot_metrics(data, delta, start, metric, ylabel):
    subset = data[(data['Delta'] == delta) & (data['Start'] == seqtime_quarter[start])]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset, x="Time", y=metric, hue="Method", marker="o")
    plt.title(f'{metric} over Time for Delta {delta}')
    plt.ylabel(ylabel)
    plt.show()

# Example usage
plot_metrics(data_tpm_reconstructed, delta=1, start=1, metric="Inertia", ylabel="Inertia")
plot_metrics(data_tpm_reconstructed, delta=1, start=1, metric="Bias", ylabel="Bias")
plot_metrics(data_tpm_reconstructed, delta=1, start=1, metric="DR", ylabel="Default Rate (DR)")

