import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class TabuSearch:
    def __init__(self, distance_matrix, tabu_tenure=10, max_iterations=1000, max_no_improve=100):
        # Recibe la matriz de distancias entre ciudades
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        
        # Parámetros del algoritmo
        self.tabu_tenure = tabu_tenure            # Cuánto tiempo un movimiento permanece prohibido
        self.max_iterations = max_iterations      # Número máximo de iteraciones
        self.max_no_improve = max_no_improve       # Número máximo de iteraciones sin mejora

        # Solución inicial aleatoria (permuta las ciudades)
        self.current_solution = list(range(self.num_cities))
        random.shuffle(self.current_solution)

        # Inicializar la mejor solución encontrada
        self.best_solution = self.current_solution.copy()
        self.best_distance = self.evaluate(self.best_solution)
        self.current_distance = self.best_distance

        # Lista tabú para registrar movimientos prohibidos (pares de índices)
        self.tabu_list = {}

        # Historial de soluciones y distancias (para analizar la evolución)
        self.history = {
            'solutions': [self.current_solution.copy()],
            'distances': [self.current_distance]
        }

    def evaluate(self, solution):
        """Calcula la distancia total de un recorrido."""
        distance = 0
        for i in range(len(solution)):
            # Suma la distancia de cada ciudad a la siguiente (última conecta con la primera)
            distance += self.distance_matrix[solution[i-1]][solution[i]]
        return distance

    def run(self, verbose=True):
        # Contadores de iteraciones
        iteration = 0
        no_improve = 0

        if verbose:
            print(f"Solución inicial: {self.current_solution}, Distancia: {self.current_distance:.2f}")

        # Bucle principal de Tabu Search
        while iteration < self.max_iterations and no_improve < self.max_no_improve:
            candidate_moves = []

            # Generar todos los vecinos posibles (intercambio de dos ciudades)
            for i in range(self.num_cities):
                for j in range(i+1, self.num_cities):
                    move = (i, j)
                    is_tabu = move in self.tabu_list and self.tabu_list[move] > iteration

                    # Crear el vecino intercambiando dos ciudades
                    neighbor = self.current_solution.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                    # Evaluar la distancia del vecino
                    neighbor_distance = self.evaluate(neighbor)

                    # Criterio de aspiración: aceptar movimientos tabú si mejoran la mejor solución conocida
                    if not is_tabu or neighbor_distance < self.best_distance:
                        candidate_moves.append((move, neighbor, neighbor_distance))

            # Si no hay vecinos válidos, terminar
            if not candidate_moves:
                if verbose:
                    print("No hay movimientos candidatos disponibles. Terminando.")
                break

            # Elegir el mejor vecino (mínima distancia)
            best_move, best_neighbor, best_neighbor_distance = min(candidate_moves, key=lambda x: x[2])

            # Actualizar solución actual
            self.current_solution = best_neighbor
            self.current_distance = best_neighbor_distance

            # Agregar el movimiento a la lista tabú
            self.tabu_list[best_move] = iteration + self.tabu_tenure
            inverse_move = (best_move[1], best_move[0])
            self.tabu_list[inverse_move] = iteration + self.tabu_tenure

            # Actualizar mejor solución si hay mejora
            if best_neighbor_distance < self.best_distance:
                self.best_solution = best_neighbor.copy()
                self.best_distance = best_neighbor_distance
                no_improve = 0
                if verbose:
                    print(f"Iteración {iteration}: Nueva mejor solución encontrada! Distancia: {self.best_distance:.2f}")
            else:
                no_improve += 1

            # Guardar en historial
            self.history['solutions'].append(self.current_solution.copy())
            self.history['distances'].append(self.current_distance)

            # Eliminar movimientos expirados de la lista tabú
            expired_moves = [move for move, exp_iter in self.tabu_list.items() if exp_iter <= iteration]
            for move in expired_moves:
                del self.tabu_list[move]

            # Incrementar iteración
            iteration += 1

        if verbose:
            print(f"\nBúsqueda Tabú terminada después de {iteration} iteraciones.")
            print(f"Mejor recorrido: {self.best_solution}")
            print(f"Distancia mínima: {self.best_distance:.2f}")

        return self.best_solution, self.best_distance, self.history

# Función para dibujar el grafo de ciudades y resaltar la mejor solución
def dibujar_grafo(coords, distance_matrix, best_solution):
    G = nx.Graph()

    # Agregar nodos con sus posiciones
    for i, (x, y) in enumerate(coords):
        G.add_node(i, pos=(x, y))

    # Agregar todas las aristas (con sus distancias)
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            G.add_edge(i, j, weight=round(distance_matrix[i][j], 1))

    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(10, 8))

    # Dibujar todos los bordes en gris claro
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=600, edge_color='lightgray', font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='gray', font_size=8)

    # Dibujar la mejor ruta encontrada (camino rojo)
    path_edges = [(best_solution[i-1], best_solution[i]) for i in range(len(best_solution))]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)

    plt.title("Grafo de Ciudades y Mejor Recorrido (en rojo)", fontsize=16)
    plt.axis('off')
    plt.show()

# Función principal para configurar el problema y ejecutar la búsqueda tabú
def ejemplo_tsp():
    np.random.seed(42)
    random.seed(42)

    # Crear una instancia de problema TSP de 4 ciudades
    num_cities = 4
    coords = np.random.rand(num_cities, 2) * 100  # Coordenadas en un plano (0-100)
    distance_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)

    print("Matriz de distancias:")
    print(distance_matrix.round(2))

    # Crear y ejecutar la búsqueda tabú
    tsp = TabuSearch(distance_matrix, tabu_tenure=5, max_iterations=500, max_no_improve=50)
    mejor_solucion, mejor_distancia, historial = tsp.run(verbose=True)

    print("\nRecorrido final:")
    print(mejor_solucion)
    print(f"Distancia final: {mejor_distancia:.2f}")
    
    # Dibujar el grafo de ciudades y la mejor ruta encontrada
    dibujar_grafo(coords, distance_matrix, mejor_solucion)

# Ejecutar si se corre este archivo directamente
if __name__ == "__main__":
    ejemplo_tsp()
