#include <iostream>
#include <tensorflow/c/c_api.h>
#include <vector>
#include "Tensor2d.h"
#include "Graph.h"

using namespace std;

int main() {
    Graph graph;

    //values of x and y for A = 12, b = -33
    vector<float> xs = {-49,-32,46,-12,18,40,24,29,18,24,12,-41,-39,-24,45,30,-6,-27,-20,21,0,25,14,-11,-22,9,46,-7,-37,-34,-18,-7,30,-10,44,11,36,35,19,24,-29,-28,23,-16,13,-1,21,6,-24,20};

    vector<float> ys = {-623,-416,521,-176,186,448,253,318,180,254,112,-524,-500,-321,505,330,-107,-359,-273,220,-30,267,133,-167,-297,76,519,-117,-475,-444,-252,-119,324,-152,492,101,397,387,192,255,-384,-370,246,-226,123,-45,222,37,-321,209,};

    auto A = graph.make_variable<TF_FLOAT>("A", 1, 1);
    auto b = graph.make_variable<TF_FLOAT>("b", 1, 1);
    auto x = graph.make_placeholder<TF_FLOAT>("x", 1, 1);
    auto y = graph.make_placeholder<TF_FLOAT>("y", 1, 1);

    Tensor2d gamma_tensor(1, 1);
    gamma_tensor(0, 0) = 0.0001;
    auto gamma = graph.make_constant<TF_FLOAT>("gamma", gamma_tensor);

    auto Ax = graph.make_matmul("Ax", A, x);
    auto Ax_plus_b = graph.make_addition("Ax_plus_b", Ax, b);
    auto Ax_plus_b_minus_y = graph.make_substraction("Ax_plus_b_minus_y", Ax_plus_b, y);
    auto Ax_plus_b_minus_y_squared = graph.make_square("Ax_plus_b_minus_y_squared", Ax_plus_b_minus_y);

    auto grads = graph.make_gradient({Ax_plus_b_minus_y_squared}, {A, b});

    auto delta_A = graph.make_mul("delta_A", grads[0], gamma);
    auto delta_b = graph.make_mul("delta_b", grads[1], gamma);

    auto update_A = graph.make_assign_sub("update_A", A, delta_A);
    auto update_b = graph.make_assign_sub("update_b", b, delta_b);

    Tensor2d A_init(1, 1);
    A_init(0, 0) = -2;
    TF_Operation* A_init_op = graph.make_variable_init("A_init", A, A_init);

    Tensor2d b_init(1, 1);
    b_init(0, 0) = 3;
    TF_Operation* b_init_op = graph.make_variable_init("b_init", b, b_init);

    graph.run_session(
            {},
            {},
            {},
            {A_init_op, b_init_op}
            );

    vector<Tensor2d> inputs;
    inputs.push_back(Tensor2d(1, 1));
    inputs.push_back(Tensor2d(1, 1));
    for (int i = 0; i < xs.size() * 500; ++i) {
        inputs[0](0, 0) = xs[i % xs.size()];
        inputs[1](0, 0) = ys[i % xs.size()];
        vector<Tensor2d> new_values = graph.run_session(
                {x, y},
                inputs,
                {A, b},
                {update_A, update_b}
                );
        cout << "Iteration " << i << ": A = " << new_values[0](0, 0) << ", b = " << new_values[1](0, 0) << "\n";
    }
    return 0;
}
