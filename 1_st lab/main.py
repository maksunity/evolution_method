import random

# Конфигурация алгоритма
POPULATION_SIZE = 100  # Размер начальной популяции
MAX_DEVIATION = 0.001  # Допустимое отклонение от базовой точки
MUTATION_RANGE = 0.05  # Амплитуда мутаций

# Инициализация начальной популяции
population = [[random.uniform(0.5, 2.0), random.uniform(0.5, 2.0)] for _ in range(POPULATION_SIZE)]
generation_count = 0

while True:
    generation_count += 1
    offspring = []
    for _ in range(POPULATION_SIZE):
        parent_x = random.choice(population)
        parent_y = random.choice(population)

        # Мутации
        mutated_x = parent_x[0] + random.uniform(-MUTATION_RANGE, MUTATION_RANGE)
        mutated_y = parent_y[1] + random.uniform(-MUTATION_RANGE, MUTATION_RANGE)

        offspring.append([
            max(mutated_x, 0.5),
            max(mutated_y, 0.5)
        ])

    combined = population + offspring
    print(f"\nПоколение {generation_count}: всего особей {len(combined)}")

    # Оценка приспособленности
    evaluated = []
    solution_found = False
    for x, y in combined:
        fitness = 3 * x ** 2 + 2 * y ** 2
        distance = abs(x - 0.5) + abs(y - 0.5)
        evaluated.append((fitness, distance, (x, y)))

        if distance < MAX_DEVIATION:
            print(f"Найдено решение: ({x:.4f}, {y:.4f}) на поколении {generation_count}")
            population = [(x, y)]
            solution_found = True
            break

    if solution_found:
        break

    evaluated.sort(key=lambda x: (-x[0], -x[1]))  # Сортировка для отбора
    survivors = [item[2] for item in evaluated[len(evaluated) // 2:]]

    population = survivors
    print(f"Сохранили {len(population)} особей после селекции")

print("\nИтоговое решение:")
print(f"x = {population[0][0]:.5f}, y = {population[0][1]:.5f}")
print(f"Потребовалось поколений: {generation_count}")