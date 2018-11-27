#include <iostream>
#include <tensorflow/c/c_api.h>
#include <vector>
#include "Tensor.h"
#include "Graph.h"
#include <numeric>
using namespace std;

int main() {
    Graph graph;

    //values of x and y for A = 12, b = -33
    vector<float> xs = {-49,-32,46,-12,18,40,24,29,18,24,12,-41,-39,-24,45,30,-6,-27,-20,21,0,25,14,-11,-22,9,46,-7,-37,-34,-18,-7,30,-10,44,11,36,35,19,24,-29,-28,23,-16,13,-1,21,6,-24,20};

    vector<float> ys = {-623,-416,521,-176,186,448,253,318,180,254,112,-524,-500,-321,505,330,-107,-359,-273,220,-30,267,133,-167,-297,76,519,-117,-475,-444,-252,-119,324,-152,492,101,397,387,192,255,-384,-370,246,-226,123,-45,222,37,-321,209,};

    TF_Output A = graph.get_output(graph.make_variable("A", {1}, TF_FLOAT), 0);
    TF_Output b = graph.get_output(graph.make_variable("b", {1}, TF_FLOAT), 0);
    TF_Output x = graph.get_output(graph.make_placeholder("x", {1}, TF_FLOAT), 0);
    TF_Output y = graph.get_output(graph.make_placeholder("y", {1}, TF_FLOAT), 0);

    Tensor gamma_tensor({1}, TF_FLOAT);
    gamma_tensor.at<TF_FLOAT>({0}) = 0.0001;
    TF_Output gamma = graph.get_output(graph.make_constant("gamma", gamma_tensor), 0);

    TF_Output Ax = graph.get_output(graph.make_mul("Ax", A, x), 0);
    TF_Output Ax_plus_b = graph.get_output(graph.make_addition("Ax_plus_b", Ax, b), 0);
    TF_Output Ax_plus_b_minus_y = graph.get_output(graph.make_substraction("Ax_plus_b_minus_y", Ax_plus_b, y), 0);
    TF_Output Ax_plus_b_minus_y_squared = graph.get_output(graph.make_square("Ax_plus_b_minus_y_squared", Ax_plus_b_minus_y), 0);

    vector<TF_Output> grads = graph.make_gradient({Ax_plus_b_minus_y_squared}, {A, b});

    TF_Output delta_A = graph.get_output(graph.make_mul("delta_A", grads[0], gamma), 0);
    TF_Output delta_b = graph.get_output(graph.make_mul("delta_b", grads[1], gamma), 0);

    TF_Operation *update_A = graph.make_assign_sub("update_A", A, delta_A);
    TF_Operation *update_b = graph.make_assign_sub("update_b", b, delta_b);

    Tensor A_init({1}, TF_FLOAT);
    A_init.at<TF_FLOAT>({0}) = -2;
    TF_Operation* A_init_op = graph.make_variable_init("A_init", A, A_init);

    Tensor b_init({1}, TF_FLOAT);
    b_init.at<TF_FLOAT>({0}) = 3;
    TF_Operation* b_init_op = graph.make_variable_init("b_init", b, b_init);

    graph.run_session(
            {},
            {},
            {},
            {A_init_op, b_init_op}
            );

    vector<Tensor> inputs;
    inputs.push_back(Tensor({1}, TF_FLOAT));
    inputs.push_back(Tensor({1}, TF_FLOAT));
    for (int i = 0; i < xs.size() * 500; ++i) {
        inputs[0].at<TF_FLOAT>({0}) = xs[i % xs.size()];
        inputs[1].at<TF_FLOAT>({0}) = ys[i % xs.size()];
        vector<Tensor> new_values = graph.run_session(
                {x, y},
                inputs,
                {A, b},
                {update_A, update_b}
                );
        cout << "Iteration " << i << ": A = " << new_values[0].at<TF_FLOAT>({0}) << ", b = " << new_values[1].at<TF_FLOAT>({0}) << "\n";
    }
    return 0;
}
