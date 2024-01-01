#ifndef MONTECARLO_H
#define MONTECARLO_H

#include <vector>
#include <thread>
#include <functional>
#include <random>

class MonteCarlo {
public:
    MonteCarlo(int num_samples, int num_threads)
        : num_samples(num_samples), num_threads(num_threads) {}

    // Example of a Monte Carlo simulation function
    double simulate(std::function<double()> func) {
        std::vector<std::thread> threads;
        double total = 0.0;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&total, this, &func, i]() {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0.0, 1.0);

                double partial_sum = 0.0;
                for (int j = 0; j < num_samples / num_threads; ++j) {
                    partial_sum += func();
                }

                total += partial_sum;
            });
        }

        for (auto& th : threads) {
            th.join();
        }

        return total / num_samples;
    }

private:
    int num_samples;
    int num_threads;
};

#endif // MONTECARLO_H
