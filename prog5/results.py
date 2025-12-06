a = ['SNR_dB', 'uncoded_theory', 'uncoded_sim', 'hamming_sim',]
b = [4, 1.25008e-02, 1.25010e-02, 1.60474e-02, ]
c = [6, 2.38829e-03, 2.38965e-03, 2.32790e-03, ]
d = [8, 1.90908e-04, 1.90971e-04, 1.17019e-04, ]
e = [10, 3.87211e-06, 3.88500e-06, 1.19610e-06, ]
f = [12, 9.00601e-09, 1.07000e-08, 3.00000e-10]

data = {}
for i in range(len(a)):
    data[a[i]] = []
    for l in [b, c, d, e, f]:
        print(i)
        print(l)
        data[a[i]].append(l[i])

print(data)
