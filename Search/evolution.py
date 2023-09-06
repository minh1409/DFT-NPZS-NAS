import random
import numpy as np
from .archencoding import search_space_type_dict
from .individiual import Individual, round_to_list

def binary_tournament_selection(population):
    individual1 = random.choice(population)
    individual2 = random.choice(population)
    while individual2 == individual1:  # Ensure we have two unique individuals for comparison
        individual2 = random.choice(population)
    if individual1.rank < individual2.rank or (individual1.rank == individual2.rank and individual1.crowding_distance > individual2.crowding_distance):
        return individual1
    else:
        return individual2

def crossover_GA(parent1, parent2, types):
    GA_cross_over = 0.5
    GA_mutatation_rate = 0.8  # Mutation rate for GA

    child1 = Individual(np.copy(parent1.decision_variables))
    child2 = Individual(np.copy(parent2.decision_variables))

    for i, type_ in enumerate(types): # duyệt qua từng chiều
        if type_ in ['search_space_block_list', 'search_space_kernel_size', 'search_space_strides']:  # GA operation
            if random.random() < GA_cross_over:
            # GA crossover
                child1.decision_variables[i] = parent2.decision_variables[i]
                child2.decision_variables[i] = parent1.decision_variables[i]
            # GA mutation
            if random.random() < GA_mutatation_rate:
                child1.decision_variables[i] = np.random.choice([0,1]) if type_ == 'search_space_block_list' else np.random.choice(search_space_type_dict[type_])
            if random.random() < GA_mutatation_rate:
                child2.decision_variables[i] = np.random.choice([0,1]) if type_ == 'search_space_block_list' else np.random.choice(search_space_type_dict[type_])
    return child1, child2

def DE_operator(parent, population, types):
    F = 0.8  # Scaling factor for DE mutation
    DE_crossover_rate = 0.8  # Crossover rate for DE

    child = Individual(np.copy(parent.decision_variables))

    rand_child_ind1 = np.random.choice([x for x in population if x not in [child]])
    rand_child_ind2 = np.random.choice([x for x in population if x not in [child, rand_child_ind1]])
    rand_child_ind3 = np.random.choice([x for x in population if x not in [child, rand_child_ind1, rand_child_ind2]])

    for i, type_ in enumerate(types): # duyệt qua từng chiều
        if type_ not in ['search_space_block_list', 'search_space_kernel_size', 'search_space_strides']:  # GA operation
            if random.random() < DE_crossover_rate:
                mutant = rand_child_ind1.decision_variables[i] + F * (rand_child_ind2.decision_variables[i] - rand_child_ind3.decision_variables[i])
                child.decision_variables[i] = round_to_list(mutant, search_space_type_dict[type_])
    return child

def generate_offspring_DE(population, types):
    offspring = []

    while len(offspring) < len(population): #GA
        client1 = binary_tournament_selection(population)
        client2 = binary_tournament_selection(population)
        child1, child2 = crossover_GA(client1, client2, types)
        offspring.extend([child1, child2])
    offspring_DE = []
    for ind in offspring[:len(population)]: #DE
        child = DE_operator(ind, offspring, types)
        offspring_DE.append(child)
    return offspring_DE[:len(population)]  # Ensure the offspring size same as the parent population size

def non_dominated_sorting(pool_population):
    # Compute Pareto rank for each individual in the population
    population = pool_population.copy()
    num_objectives = len(population[0].objective_values)

    for individual in population:
        individual.dominates = []  # List of solutions dominated by this individual
        individual.dominated_count = 0  # Count of solutions that dominate this individual

    for i in range(len(population)):
        for j in range(i+1, len(population)):
            dominance = sum(population[i].objective_values[k] >= population[j].objective_values[k] for k in range(num_objectives))
            if dominance == num_objectives:  # i dominates j
                population[i].dominates.append(population[j])
                population[j].dominated_count += 1
            elif dominance == 0:  # j dominates i
                population[j].dominates.append(population[i])
                population[i].dominated_count += 1

    # Assign ranks based on domination
    pareto_fronts = []
    i = 0
    while population:
        pareto_fronts.append([])
        for individual in population[:]:
            if individual.dominated_count == 0:
                individual.rank = i
                pareto_fronts[i].append(individual)
                population.remove(individual)
        for individual in pareto_fronts[i]:
            for dominated in individual.dominates:
                dominated.dominated_count -= 1
        i += 1
    return pareto_fronts


def crowding_distance_assignment(population):
    # Get unique ranks in the population
    unique_ranks = np.unique([ind.rank for ind in population])

    # Calculate crowding distance for each rank
    for rank in unique_ranks:
        # Get individuals with the current rank
        front = [ind for ind in population if ind.rank == rank]
        front_size = len(front)
        if front_size == 0:
            continue
        elif front_size == 1:
            front[0].crowding_distance = np.inf
            continue

        # Initialize crowding distance
        for ind in front:
            ind.crowding_distance = 0

        # Calculate crowding distance in each objective
        num_objectives = len(front[0].objective_values)
        for m in range(num_objectives):
            front.sort(key=lambda x: x.objective_values[m])

            # Set boundaries to infinity
            front[0].crowding_distance = front[-1].crowding_distance = np.inf

            # Calculate crowding distance
            for i in range(1, front_size - 1):
                front[i].crowding_distance += (front[i+1].objective_values[m] - front[i-1].objective_values[m])


def select_next_population(population, pop_size):
    population.sort(key=lambda x: (x.rank, -x.crowding_distance))
    return population[:pop_size]
