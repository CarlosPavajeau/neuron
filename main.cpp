#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include "neuron.h"

void
train_neuron(neuron &neuron, const std::vector<std::vector<float>> &inputs, const std::vector<float> &outputs,
             int max_steps, const std::string &training_type);

int main() {
    std::vector<std::vector<float>> and_or_inputs = {
            {1.0f, 1.0f},
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {0.0f, 0.0f}
    };
    std::vector<std::vector<float>> and_variant_inputs = {
            {1.0f, 1.0f, 1.0f},
            {1.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 0.0f}
    };
    std::vector<float> and_outputs = {
            1.0f, 0.0f, 0.0f, 0.0f
    };
    std::vector<float> or_outputs = {
            1.0f, 1.0f, 1.0f, 0.0f
    };

    neuron and_neuron(2, 0.03);
    neuron or_neuron(2, 0.03);
    neuron and_variant(3, 0.03);


    train_neuron(and_neuron, and_or_inputs, and_outputs, 100, "And training");
    train_neuron(or_neuron, and_or_inputs, or_outputs, 100, "Or training");
    train_neuron(and_variant, and_variant_inputs, and_outputs, 100, "And variant training");


    std::cout << "--------------------TEST AND-----------------------" << std::endl;
    printf("Test E1: 1, E2: 1 -> %f\n", and_neuron.output({1.0f, 1.0f}));
    printf("Test E1: 0, E2: 1 -> %f\n", and_neuron.output({0.0f, 1.0f}));
    printf("Test E1: 1, E2: 0 -> %f\n", and_neuron.output({1.0f, 0.0f}));
    printf("Test E1: 0, E2: 0 -> %f\n", and_neuron.output({0.0f, 0.0f}));

    std::cout << "--------------------TEST OR-----------------------" << std::endl;
    printf("Test E1: 1, E2: 1 -> %f\n", or_neuron.output({1.0f, 1.0f}));
    printf("Test E1: 0, E2: 1 -> %f\n", or_neuron.output({0.0f, 1.0f}));
    printf("Test E1: 1, E2: 0 -> %f\n", or_neuron.output({1.0f, 0.0f}));
    printf("Test E1: 0, E2: 0 -> %f\n", or_neuron.output({0.0f, 0.0f}));

    std::cout << "--------------------TEST AND VARIANT-----------------------" << std::endl;
    printf("Test E1: 1, E2: 1, E3: 1 -> %f\n", and_variant.output({1.0f, 1.0f, 1.0f}));
    printf("Test E1: 1, E2: 1, E3: 0 -> %f\n", and_variant.output({1.0f, 1.0f, 0.0f}));
    printf("Test E1: 0, E2: 0, E3: 1 -> %f\n", and_variant.output({0.0f, 0.0f, 0.0f}));
    printf("Test E1: 0, E2: 0, E3: 0 -> %f\n", and_variant.output({0.0f, 0.0f, 0.0f}));
    printf("Test E1: 0, E2: 1, E3: 0 -> %f\n", and_variant.output({0.0f, 1.0f, 0.0f}));


    return 0;
}

void
train_neuron(neuron &neuron, const std::vector<std::vector<float>> &inputs, const std::vector<float> &outputs,
             int max_steps,
             const std::string &training_type) {
    std::cout << "-----------------------" << training_type << "--------------------" << std::endl;
    bool sw = false;
    int steps = 0;

    while (!sw && (steps <= max_steps)) {
        ++steps;
        sw = true;

        std::cout << "---------------------------------------------" << std::endl;
        printf("Weight 1: %f\n", neuron.get_weight(0));
        printf("Weight 2: %f\n", neuron.get_weight(1));
        printf("Sill: %f\n", neuron.get_sill());
        printf("Steps: %d\n", steps);
        printf("Error: %f\n", neuron.get_error());

        for (int i = 0; i < inputs.size(); ++i) {
            auto input = inputs[i];
            float result = neuron.output(input);

            for (int j = 0; j < input.size(); ++j) {
                printf("E%d: %f", j + 1, input[j]);
                if (j < input.size() - 1) {
                    printf(", ");
                }
            }
            printf(" -> %f\n", result);

            if (result != outputs[i]) {
                neuron.learn(input, outputs[i]);
                sw = false;
            }
        }
    }
}
