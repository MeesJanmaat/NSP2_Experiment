import matplotlib.pyplot as plt

E = {
    0: -302.074,
    0.5: -265.963,
    1: -229.852,
    1.1: -187.4925,
    1.5: -151.3815,
    2: -72.911,
}

for e1 in E.keys():
    for e2 in E.keys():
        if e1 != e2:
            diff = abs(E[e1] - E[e2])
            print(f"{e1} --> {e2}: {diff}")
            plt.scatter(diff, 1)

plt.show()
