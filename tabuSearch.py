import random
import copy

def calcular_distancia(ruta, distancias):
    distancia_total = 0
    for i in range(len(ruta)):
        distancia_total += distancias[ruta[i]][ruta[(i + 1) % len(ruta)]]
    return distancia_total

def generar_vecinos(ruta):
    vecinos = []
    for i in range(len(ruta)):
        for j in range(i + 1, len(ruta)):
            vecino = ruta[:]
            vecino[i], vecino[j] = vecino[j], vecino[i]
            vecinos.append(vecino)
    return vecinos

def tabu_search(distancias, ciudades, iteraciones=100, tamaño_tabu=10):
    ruta_actual = ciudades[:]
    random.shuffle(ruta_actual)
    mejor_ruta = ruta_actual[:]
    mejor_distancia = calcular_distancia(mejor_ruta, distancias)

    lista_tabu = []

    for iteracion in range(iteraciones):
        vecinos = generar_vecinos(ruta_actual)
        vecinos = [v for v in vecinos if v not in lista_tabu]

        if not vecinos:
            break

        mejor_vecino = min(vecinos, key=lambda x: calcular_distancia(x, distancias))
        mejor_vecino_distancia = calcular_distancia(mejor_vecino, distancias)

        if mejor_vecino_distancia < mejor_distancia:
            mejor_ruta = mejor_vecino[:]
            mejor_distancia = mejor_vecino_distancia

        lista_tabu.append(mejor_vecino)
        if len(lista_tabu) > tamaño_tabu:
            lista_tabu.pop(0)

        ruta_actual = mejor_vecino

        print(f"Iteración {iteracion+1}: Distancia = {mejor_vecino_distancia}")

    return mejor_ruta, mejor_distancia

# 🔧 Ejemplo de uso
ciudades = list(range(5))
distancias = [
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0]
]

mejor_ruta, mejor_distancia = tabu_search(distancias, ciudades, iteraciones=100, tamaño_tabu=5)
print(f"\nMejor ruta encontrada: {mejor_ruta}")
print(f"Distancia total: {mejor_distancia}")
