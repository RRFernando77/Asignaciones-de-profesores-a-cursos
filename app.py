import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns

# Preferencias de los profesores por los cursos
preferencias = np.array([
    [5, 8, 5, 9, 7],  # Preferencias del profesor A
    [8, 2, 10, 7, 9],  # Preferencias del profesor B
    [5, 3, 8, 9, 9],  # Preferencias del profesor C
    [9, 6, 9, 7, 10],  # Preferencias del profesor D
    [7, 8, 8, 8, 5]   # Preferencias del profesor E
])

# Número de profesores y cursos
num_profesores, num_cursos = preferencias.shape

# Vector de costes (convertido a negativo para maximizar con linprog)
costes = -preferencias.flatten()

# Matrices de restricciones de igualdad y desigualdad
A_eq = np.zeros((num_cursos, num_profesores * num_cursos))
b_eq = np.ones(num_cursos)

for i in range(num_cursos):
    A_eq[i, i::num_cursos] = 1

A_ub = np.zeros((num_profesores, num_profesores * num_cursos))
b_ub = np.ones(num_profesores)

for i in range(num_profesores):
    A_ub[i, i*num_cursos:(i+1)*num_cursos] = 1

# Límites para las variables de decisión
x_bounds = [(0, 1) for _ in range(num_profesores * num_cursos)]

# Resolver el problema de programación lineal
res = linprog(c=costes, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='simplex')
    
# Mostrar los resultados
asignaciones = res.x.reshape(num_profesores, num_cursos)
valor_optimo = -res.fun

# Función para mostrar el resultado
def print_results(asignaciones, valor_optimo):
    print("Solución óptima encontrada:\n")
    print("Asignaciones óptimas de profesores a cursos:")

    for i in range(num_profesores):
        for j in range(num_cursos):
            if asignaciones[i, j] > 0.99:
                print(f"El profesor {chr(65+i)} enseña el curso C{j+1}")

    print(f"\nValor óptimo de las preferencias: {valor_optimo}\n")

    print("Comprobación de restricciones:")
    for j in range(num_cursos):
        total_asignaciones = sum(asignaciones[:, j])
        print(f"Total de asignaciones para el curso C{j+1}: {total_asignaciones} (Debe ser 1)")

    for i in range(num_profesores):
        total_asignaciones = sum(asignaciones[i, :])
        print(f"Total de asignaciones para el profesor {chr(65+i)}: {total_asignaciones} (Debe ser <= 1)")

    print("\nDetalles del problema:")

# Imprimir resultados
print_results(asignaciones, valor_optimo)

# Detalles del problema
funcion_objetivo = ""
for i in range(num_profesores):
    for j in range(num_cursos):
        funcion_objetivo += f"{preferencias[i, j]}*x_{chr(65+i)}{j+1} + "

funcion_objetivo = funcion_objetivo.strip(" + ")
print("\nFunción Objetivo (Maximizar preferencias):")
print(funcion_objetivo)

print("\nRestricciones de igualdad (cada curso asignado a un profesor):")
for j in range(num_cursos):
    restriccion = " + ".join([f"x_{chr(65+i)}{j+1}" for i in range(num_profesores)]) + " = 1"
    print(restriccion)

print("\nRestricciones de desigualdad (cada profesor a lo más un curso):")
for i in range(num_profesores):
    restriccion = " + ".join([f"x_{chr(65+i)}{j+1}" for j in range(num_cursos)]) + " <= 1"
    print(restriccion)

# Visualización de las asignaciones
plt.figure(figsize=(10, 6))
ax = sns.heatmap(asignaciones, annot=True, fmt=".1f", cmap="Blues", xticklabels=[f'C{j+1}' for j in range(num_cursos)], yticklabels=[chr(65+i) for i in range(num_profesores)])
plt.title("Asignaciones de Profesores a Cursos")
plt.xlabel("Cursos")
plt.ylabel("Profesores")
plt.show()
