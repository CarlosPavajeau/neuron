//
// Created by cantte on 16/03/21.
//

#ifndef PERCEPTRON_NEURON_H
#define PERCEPTRON_NEURON_H


#include <vector>

class neuron {
public:
    explicit neuron(int inputsNumber, float training_rate = 0.3);

    void learn();
    void learn(std::vector<float> inputs, float expectedOutput);

    float output(std::vector<float> inputs);

    [[nodiscard]] float get_weight(int weight) const;
    [[nodiscard]] float get_sill() const;
    [[nodiscard]] float get_error() const;
private:
    std::vector<float> _weights;
    std::vector<float> _previous_weights;
    std::vector<float> _errors;
    float _sill = 0.0f;
    float _training_rate;
    float _previous_sill = 0.0f;

    static float predict(float input);
    float next_input(std::vector<float> inputs);
};


#endif //PERCEPTRON_NEURON_H
