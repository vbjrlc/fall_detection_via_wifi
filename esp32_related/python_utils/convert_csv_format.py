import math
import numpy as np
import csv
import time

# Define names of columns where are the useful datas
column_data = 'CSI_DATA'
column_time = 'real_timestamp'

# ESP32 collects data with communication frames that must be removed
begin = 12
end = 118

# Wanted length of data matrix
size = 90

#Parameters
x1 = 37
x2 = 16

# Collect the data and time matrices from the .csv file generated by ESP32
def collect_data(source):
    # Arrays definition
    d = []
    t = []
    try:
        with open(source, 'r', newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            if column_data in csv_reader.fieldnames and column_time in csv_reader.fieldnames:
                for line in csv_reader:
                    d.append(line[column_data])
                    t.append(float(line[column_time]))
            else:
                print(f"Column(s) '{column_data}' and/or '{column_time}' not found.")
    except FileNotFoundError:
        print(f"File '{source}' not found.")
    return d, t

# Re arrange time vector to turn it into a matrix
def manip_time(t):
    h = len(t)
    time2 = np.zeros((h, 1))
    for i in range(h):
        time2[i][0] = t[i]
    return time2

# Re arrange data matrix to give it the wanted format
def convert_to_matrix(l):
    return list(map(int, l.strip('[]').split()))

# Calculate the amplitude, based on complex number CSI 
def ampl_calc(l):
    imaginary = []
    real = []
    # CSI data list consist of real and imaginqry numbers put one after the other
    for i, val in enumerate(l):
        if i % 2 == 0:
            imaginary.append(val)
        else:
            real.append(val)
    size = len(l)
    amplitudes = []
    # Calculate the amplitude with basic formula
    if len(imaginary) > 0 and len(real) > 0:
        for j in range(int(size / 2)):
            amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
            amplitudes.append(amplitude_calc)
    return amplitudes

# Calculate the amplitude for the whole data matrix
def ampl_calc_matrix(m):
    h, w = len(m), len(m[0])
    res = np.zeros((h, int(w/2)))
    for k in range(h):
        row = m[k]
        amplitudes = ampl_calc(row)
        res[k] = amplitudes
    return res

# Fix the following bug: when calculating the amplitude, for each sample, the 27th term is always '0',
# which is not normal. To fix the bug, we take instead the mean of 26th and 28th terms.
def fix_ampl_bug(m):
    for k in range(len(m)):
        m[k][26] = (m[k][25] + m[k][27])/2

# Remove communication frames
def rm_unwanted_data(m):
    h = len(m)
    res = np.zeros((h, end-begin))
    for i in range(h):
        for j in range(begin, end):
            res[i][j-begin] = m[i][j]
    return res

# Up the matrix size to make it the same as the AI wants by adding the mean of two close values
# until reaching wanted size.
def add_values(m):
    h, w = len(m), len(m[0])
    res = np.zeros((h, size))
    for k in range(h):
        row = m[k]
        j2 = 0
        for j1 in range(x1):
            res[k][j2] = row[j1]
            mean = (row[j1]+row[j1+1])/2
            res[k][j2+1] = mean
            j2 += 2
        for i in range(x1, x1+x2):
            res[k][j2] = row[i]
            j2 += 1
    return res

# Save data in the wanted format: no headers, time vector in first column, data matrix after
def save_as_csv(d, t, dest):
    try:
        with open(dest, 'w', newline='') as csv_file:
            nb_col = len(d[0])
            fieldnames = ['t'] + [f'd{i}' for i in range(1, nb_col+1)]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            h = len(d)
            if h != len(t):
                print("Error with lengths of time and data vectors.")
                return
            for i in range(h):
                row_data = {'t': t[i][0]}
                for j in range(len(d[i])):
                    row_data[f'd{j+1}'] = d[i][j]
                csv_writer.writerow(row_data)
        csv_file.close()
    except FileNotFoundError:
        print(f"File '{dest}' not found.")

# Main
def main_c(src, dest):
    t0 = time.time()
    data, time1 = collect_data(src)
    time2 = manip_time(time1)
    matrix_data = [convert_to_matrix(l) for l in data]
    final_data = rm_unwanted_data(matrix_data)
    ampl = ampl_calc_matrix(final_data)
    fix_ampl_bug(ampl)
    ampl2 = add_values(ampl)
    save_as_csv(ampl2, time2, dest)
    t1 = time.time()
    tt = t1-t0
    print(f"File '{src}' converted successfully in {tt}s.")


source10 = "../sauv_results/fall.csv"
source11 = "../sauv_results/fall2.csv"
dest10 = "../converted_data/fall.csv"
dest11 = "../converted_data/fall2.csv"

main_c(source10, dest10)
main_c(source11, dest11)