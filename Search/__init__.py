from .individiual import initialize_population, evaluate_paretos, evaluate_population, EVALUATON_DICT,Individual
from .evolution import generate_offspring_DE, non_dominated_sorting, crowding_distance_assignment, select_next_population
from .archencoding import encode_str_structure
import pickle
import os
def nsga2_DE(measurer, representative_params, types, pop_size, num_generations, max_model_size, max_layers, save_dir, freq_print=1, freq_save = 1):
    # Initialize population
    print("Initialize population")
    population = initialize_population(representative_params, pop_size, types, max_model_size, max_layers)
    pareto_fronts = []

    # Evaluate population
    population = evaluate_population(measurer, representative_params, population, max_model_size, max_layers, save_dir)

    for i in range(num_generations):
        print(f"Iteration {i+1} of {num_generations}")

        # Generate offspring
        offspring_population = generate_offspring_DE(population, types)
        # Evaluate offspring
        offspring_population = evaluate_population(measurer, representative_params, offspring_population, max_model_size, max_layers, save_dir)
        # Combine parent and offspring population
        combined_population = population + offspring_population

        # Non-dominated sorting
        pareto_fronts = non_dominated_sorting(combined_population)
        if (i+1)%freq_print==0:
            print("Current Pareto fronts:")
            for j, front in enumerate(pareto_fronts):
                print(f"Front {j+1}:")
                for individual in front:
                    print(individual.objective_values, individual.structre_str)
                if j == 0:
                    break
            print('---')

        # Crowding distance assignment
        crowding_distance_assignment(combined_population)
        # Generate next population
        population = select_next_population(combined_population, pop_size)
        if (i+1)%freq_save==0:
            full_pop = {'pareto_fronts_str': [[ind.structre_str for ind in front] for front in pareto_fronts], 'population_str': [ind.structre_str for ind in population]}
            with open(os.path.join(save_dir, 'full_pop.pickle'), 'wb') as handle:
                pickle.dump(full_pop, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pareto_fronts

def nsga2_DE_resume(measurer, representative_params, population_str, pareto_fronts_str, types, pop_size, num_generations, max_model_size, max_layers, save_dir, freq_print=1, freq_save = 1):
    # Evaluate population
    population = [Individual(encode_str_structure(representative_params, ind_str)) for ind_str in population_str]
    pareto_fronts = [[Individual(encode_str_structure(representative_params, ind_str)) for ind_str in front] for front in pareto_fronts_str]
    population = evaluate_population(measurer, representative_params, population, max_model_size, max_layers, save_dir)
    pareto_fronts = evaluate_paretos(measurer, representative_params, pareto_fronts, max_model_size, max_layers, save_dir)
    for i in range(num_generations):
        print(f"Iteration {i+1} of {num_generations}")

        # Generate offspring
        offspring_population = generate_offspring_DE(population, types)
        # Evaluate offspring
        offspring_population = evaluate_population(measurer, representative_params, offspring_population, max_model_size, max_layers, save_dir)
        # Combine parent and offspring population
        combined_population = population + offspring_population

        # Non-dominated sorting
        pareto_fronts = non_dominated_sorting(combined_population)
        if (i+1)%freq_print==0:
            print("Current Pareto fronts:")
            for j, front in enumerate(pareto_fronts):
                print(f"Front {j+1}:")
                for individual in front:
                    print(individual.objective_values, individual.structre_str)
                if j == 0:
                    break
            print('---')

        # Crowding distance assignment
        crowding_distance_assignment(combined_population)
        # Generate next population
        population = select_next_population(combined_population, pop_size)
        
        if (i+1)%freq_save==0:
            full_pop = {'pareto_fronts_str': [[ind.structre_str for ind in front] for front in pareto_fronts], 'population_str': [ind.structre_str for ind in population]}
            with open(os.path.join(save_dir, 'full_pop.pickle'), 'wb') as handle:
                pickle.dump(full_pop, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pareto_fronts