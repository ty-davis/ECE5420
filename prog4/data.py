es = ["$E_s$", 7, 7, 7, 7, 7, 7,]
n0 = ["$N_0$", 0.70000000, 0.44167014, 0.27867502, 0.17583205, 0.11094252, 0.07000000,]
es_N0 = ["$E_s / N_0$", 1.429, 2.264, 3.588, 5.687, 9.014, 14.286,]
es_N0_db = ["$E_s / N_0$ in dB", 10.0, 12, 14, 16, 18, 20,]
sim   = ["$P_e$ Simulation",  2.3157e-02, 8.3524e-03, 1.8483e-03, 1.8658e-04, 5.4100e-06, 2.0000e-08]
upper = ["$P_e$ Upper Bound", 1.3645e-01, 5.0009e-02, 1.1078e-02, 1.1169e-03, 3.2663e-05, 1.3546e-07]
lower = ["$P_e$ Lower Bound", 1.1371e-02, 4.1674e-03, 9.2314e-04, 9.3075e-05, 2.7219e-06, 1.1288e-08]


for i, row in enumerate([es, n0, es_N0, es_N0_db, sim, upper, lower]):
    for j, val in enumerate(row):
        if type(val) == str:
            print(f"{val} & ", end="")
        else:
            s = f"{val:.4g}"
            if i > 3:
                s = f"{val:.4e}"
            if j == len(row) - 1:
                print(f"{s} \\\\")
            else:
                print(f"{s} & ", end="")


