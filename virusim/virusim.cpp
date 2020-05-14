//  
// Simulação de propagação de vírus.
// Adaptada de um código proposto por David Joiner (Kean University).
//
// Uso: virusim <tamanho-da-populacao> <nro. experimentos> <probab. maxima> 

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include "Random.h"
#include "Population.h"

#include <omp.h>

void checkCommandLine(int argc, char** argv, int& threads, int& size, int& trials, int& probs)
{
	char* ptr;
	if (argc > 1)
	{
		threads = (int)strtol(argv[1], &ptr, 10);
	}
	if (argc > 2)
	{
		size = (int)strtol(argv[1], &ptr, 10);
	}
	if (argc > 3)
	{
		trials = (int)strtol(argv[2], &ptr, 10);
	}
	if (argc > 4)
	{
		probs = (int)strtol(argv[3], &ptr, 10);
	}
}

int main(int argc, char* argv[])
{

	// parâmetros dos experimentos
	int num_threads = 4;
	int population_size = 30;
	int n_trials = 5000;
	int n_probs = 101;

	double prob_min = 0.0;
	double prob_max = 1.0;
	int base_seed = 100;

	checkCommandLine(argc, argv, num_threads, population_size, n_trials, n_probs);

	time_t start_time, end_time;
	start_time = time(nullptr);
	try
	{
		/// probabilidades a serem testadas (entrada)
		auto* prob_spread = new double[n_probs];
		/// percentuais de infectados (saída)
		auto* percent_infected = new double[n_probs];

		double prob_step = (prob_max - prob_min) / (double)(n_probs - 1);

		printf("Executando com %d thread(s)...\n", num_threads);
		printf("Probabilidade, Percentual Infectado\n");
#pragma omp parallel default(none) shared(prob_spread, percent_infected, std::cerr) firstprivate(population_size, n_probs, prob_step, base_seed, n_trials, prob_min) num_threads(num_threads)
		{
			auto* population = new Population(population_size);

#pragma omp for
			// para cada probabilidade, calcula o percentual de pessoas infectadas
			for (int ip = 0; ip < n_probs; ip++)
			{
				prob_spread[ip] = prob_min + (double)ip * prob_step;
				percent_infected[ip] = 0.0;

				Random rand;
				rand.setSeed(base_seed + ip); // nova seqüência de números aleatórios

				// executa vários experimentos para esta probabilidade
				for (int it = 0; it < n_trials; it++)
				{
					// queima floresta até o fogo apagar
					population->propagateUntilOut(population->centralPerson(), prob_spread[ip], rand);
					percent_infected[ip] += population->getPercentInfected();
				}

				// calcula média dos percentuais de árvores queimadas
				percent_infected[ip] /= n_trials;

				// mostra resultado para esta probabilidade
				printf("%d: %lf, %lf (thread %d)\n", ip, prob_spread[ip], percent_infected[ip], omp_get_thread_num());
			}
		}

		delete[] prob_spread;
		delete[] percent_infected;
	}
	catch (std::bad_alloc& ex)
	{
		std::cerr << "Erro: alocacao de memoria" << std::endl;
		exit(1);
	}
	end_time = time(nullptr);
	printf("Tempo total: %.0f segundos\n", difftime(end_time, start_time));

	return 0;
}

