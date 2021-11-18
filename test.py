import matplotlib.pyplot as plt

S = 1 / 2
L = 0
I = 3 / 2
J = S + L
F = I + J

g_F = (1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))) * (
    (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1))
)
print(f"g_F = {g_F} (F=2)")

S = 1 / 2
L = 1
I = 3 / 2
J = S + L
F = I + J

g_F = (1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))) * (
    (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1))
)
print(f"g_F = {g_F} (F=3)")
