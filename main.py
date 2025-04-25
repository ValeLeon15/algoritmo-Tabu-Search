import random
import copy
import numpy as np

class TabuSearchMateriales:
    """
    Implementación de Tabu Search para la optimización de la disposición
    de materiales con el fin de maximizar el poder aislante.

    Parámetros:
    - initial_solution: permutación inicial de los materiales (lista de 7 elementos)
    - objective_function: función que evalúa el poder aislante de una disposición
    - tabu_tenure: duración de la prohibición para movimientos en la lista tabú
    - max_iterations: número máximo de iteraciones
    - max_no_improve: número máximo de iteraciones sin mejora antes de terminar
    """

    def __init__(self, initial_solution, objective_function,
                 tabu_tenure=10, max_iterations=100, max_no_improve=20):
        self.current_solution = initial_solution
        self.best_solution = initial_solution.copy()
        self.objective_function = objective_function
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve

        # Evaluar solución inicial
        self.best_value = self.objective_function(self.best_solution)
        self.current_value = self.best_value

        # Inicializar la lista tabú (para almacenar los movimientos prohibidos)
        # Formato: {(i, j): iteration_expires}
        self.tabu_list = {}

        # Historial para seguimiento
        self.history = {
            'solutions': [self.current_solution.copy()],
            'values': [self.current_value]
        }

    def run(self, verbose=True):
        """
        Ejecuta el algoritmo Tabu Search.

        Args:
            verbose: Si es True, muestra información durante la ejecución

        Returns:
            tuple: (mejor_solución, mejor_valor, historial)
        """
        iteration = 0
        no_improve = 0

        if verbose:
            print(f"Solución inicial: {self.current_solution}, Valor: {self.current_value}")

        while iteration < self.max_iterations and no_improve < self.max_no_improve:
            # Generar todos los movimientos posibles (intercambios de pares)
            candidate_moves = []
            for i in range(len(self.current_solution)):
                for j in range(i+1, len(self.current_solution)):
                    # Verificar si el movimiento está en la lista tabú
                    move = (i, j)
                    is_tabu = move in self.tabu_list and self.tabu_list[move] > iteration

                    # Realizar el movimiento (intercambio)
                    neighbor = self.current_solution.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                    # Evaluar el vecino
                    neighbor_value = self.objective_function(neighbor)

                    # Considerar el movimiento si no es tabú o satisface el criterio de aspiración
                    if not is_tabu or neighbor_value > self.best_value:
                        candidate_moves.append((move, neighbor, neighbor_value))

            # Si no hay movimientos candidatos, terminar
            if not candidate_moves:
                if verbose:
                    print("No hay movimientos candidatos disponibles. Terminando.")
                break

            # Seleccionar el mejor movimiento
            best_move, best_neighbor, best_neighbor_value = max(
                candidate_moves, key=lambda x: x[2]
            )

            # Actualizar la solución actual
            self.current_solution = best_neighbor
            self.current_value = best_neighbor_value

            # Añadir el movimiento a la lista tabú
            self.tabu_list[best_move] = iteration + self.tabu_tenure
            # También hacemos tabú el movimiento inverso
            inverse_move = (best_move[1], best_move[0])
            self.tabu_list[inverse_move] = iteration + self.tabu_tenure

            # Actualizar la mejor solución si hay mejora
            if best_neighbor_value > self.best_value:
                self.best_solution = best_neighbor.copy()
                self.best_value = best_neighbor_value
                no_improve = 0
                if verbose:
                    print(f"Iteración {iteration}: Mejora encontrada! Valor: {self.best_value}")
            else:
                no_improve += 1
                if verbose and iteration % 10 == 0:
                    print(f"Iteración {iteration}: Sin mejora durante {no_improve} iteraciones.")

            # Guardar para historial
            self.history['solutions'].append(self.current_solution.copy())
            self.history['values'].append(self.current_value)

            # Limpiar movimientos expirados de la lista tabú
            expired_moves = [move for move, exp_iter in self.tabu_list.items() if exp_iter <= iteration]
            for move in expired_moves:
                del self.tabu_list[move]

            iteration += 1

        if verbose:
            print(f"\nBúsqueda Tabú completada después de {iteration} iteraciones.")
            print(f"Mejor solución: {self.best_solution}")
            print(f"Mejor valor (poder aislante): {self.best_value}")

        return self.best_solution, self.best_value, self.history


# Ejemplo de uso con una función objetivo simulada
def ejemplo_materiales_aislantes():
    """
    Ejemplo de uso de Tabu Search para optimizar la disposición de materiales aislantes.
    """
    # Función objetivo simulada (caja negra)
    # En un caso real, esta función vendría dada externamente o mediante mediciones
    def poder_aislante(solucion):
        """
        Simula la evaluación del poder aislante de una disposición de materiales.
        En un caso real, esta sería la 'caja negra' mencionada en el problema.

        Args:
            solucion: Lista de 7 elementos representando la permutación de materiales

        Returns:
            float: Valor del poder aislante (mayor es mejor)
        """
        # Ejemplo simplificado: algunas combinaciones de materiales adyacentes
        # tienen mejor poder aislante que otras

        # Matriz de compatibilidad: compatibilidad[i][j] representa qué tan bien
        # funciona el material i cuando está adyacente al material j
        compatibilidad = np.array([
            [0.0, 0.5, 0.3, 0.7, 0.2, 0.8, 0.1],  # Material 1
            [0.5, 0.0, 0.9, 0.4, 0.6, 0.2, 0.3],  # Material 2
            [0.3, 0.9, 0.0, 0.1, 0.8, 0.7, 0.5],  # Material 3
            [0.7, 0.4, 0.1, 0.0, 0.5, 0.3, 0.9],  # Material 4
            [0.2, 0.6, 0.8, 0.5, 0.0, 0.4, 0.7],  # Material 5
            [0.8, 0.2, 0.7, 0.3, 0.4, 0.0, 0.6],  # Material 6
            [0.1, 0.3, 0.5, 0.9, 0.7, 0.6, 0.0]   # Material 7
        ])

        # Calcular el poder aislante basado en la compatibilidad de materiales adyacentes
        poder = 0.0
        for i in range(len(solucion) - 1):
            mat1 = solucion[i] - 1  # Restamos 1 porque los materiales están numerados 1-7
            mat2 = solucion[i+1] - 1
            poder += compatibilidad[mat1][mat2]

        # Añadir un factor de complejidad: algunos materiales funcionan mejor en posiciones específicas
        for i, material in enumerate(solucion):
            # Por ejemplo, el material 3 funciona mejor en los extremos
            if material == 3 and (i == 0 or i == len(solucion) - 1):
                poder += 0.5
            # El material 5 funciona mejor en el centro
            elif material == 5 and i == len(solucion) // 2:
                poder += 0.7

        return poder

    # Solución inicial: permutación aleatoria de los 7 materiales
    materiales = list(range(1, 8))  # Materiales numerados del 1 al 7
    random.shuffle(materiales)
    initial_solution = materiales

    print(f"Solución inicial: {initial_solution}")
    print(f"Poder aislante inicial: {poder_aislante(initial_solution):.4f}\n")

    # Ejecutar Tabu Search
    ts = TabuSearchMateriales(
        initial_solution=initial_solution,
        objective_function=poder_aislante,
        tabu_tenure=7,  # Duración de la prohibición
        max_iterations=100,
        max_no_improve=15
    )

    mejor_solucion, mejor_valor, historial = ts.run(verbose=True)

    # Visualizar la evolución del poder aislante
    print("\nEvolución del poder aislante durante la búsqueda:")
    for i, (solucion, valor) in enumerate(zip(historial['solutions'][:10], historial['values'][:10])):
        print(f"Iteración {i}: {solucion} -> {valor:.4f}")

    if len(historial['values']) > 10:
        print("...")
        for i in range(max(10, len(historial['values'])-5), len(historial['values'])):
            print(f"Iteración {i}: {historial['solutions'][i]} -> {historial['values'][i]:.4f}")

    print("\nResumen:")
    print(f"Mejor disposición de materiales: {mejor_solucion}")
    print(f"Poder aislante máximo alcanzado: {mejor_valor:.4f}")

# Para ejecutar el ejemplo
if __name__ == "__main__":
    ejemplo_materiales_aislantes()
