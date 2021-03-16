//
// Created by cantte on 16/03/21.
//

#include <random>
#include <utility>
#include "neuron.h"

neuron::neuron(int inputsNumber, float training_rate) {
    _previous_weights = std::vector<float>(inputsNumber);
    _training_rate = training_rate;
    _weights = std::vector<float>(inputsNumber);
    init();
}

void neuron::init() {
    std::random_device random_device;
    std::mt19937 random_engine(random_device());
    std::uniform_real_distribution<float> distribution(-1, 1);

    for (float & _previous_weight : _previous_weights) {
        _previous_weight = distribution(random_engine);
    }
    _previous_sill = distribution(random_engine);

    _weights = _previous_weights;
    _sill = _previous_sill;
}

void neuron::learn(std::vector<float> inputs, float expectedOutput) {
    float error = expectedOutput - output(inputs);
    _errors.push_back(error);
    for (int i = 0; i < _weights.size(); ++i) {
        _weights[i] = _previous_weights[i] + _training_rate * error * inputs[i];
    }

    _sill = _previous_sill + _training_rate * error;
    _previous_weights = _weights;
    _previous_sill = _sill;
}

float neuron::output(std::vector<float> inputs) {
    return predict(next_input(std::move(inputs)));
}

float neuron::predict(float input) {
    return input > 0 ? 1 : 0;
}

float neuron::next_input(std::vector<float> inputs) {
    float acc = 0.0;
    for (int i = 0; i < _weights.size(); ++i) {
        acc += inputs[i] * _weights[i];
    }

    return acc + _sill;
}

float neuron::get_weight(int weight) const {
    return _weights[weight];
}

float neuron::get_sill() const {
    return _sill;
}

float neuron::get_error() const {
    return (!_errors.empty()) ? _errors.back() : 1;
}