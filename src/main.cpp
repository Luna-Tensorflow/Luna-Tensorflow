#include <iostream>
#include <tensorflow/c/c_api.h>
#include <vector>
#include "Tensor2d.h"
#include "Graph.h"

using namespace std;

int main() {
    int n;
    cin >> n;

    Graph graph;

    vector<Tensor2d> inputs;
    inputs.push_back(Tensor2d(n, 1));
    Tensor2d input(n, 1);
    cout << "Input tensor: \n";
    for (int i = 0; i < n; ++i) {
        cin >> inputs[0](i, 0);
    }

    auto A = graph.make_variable<TF_FLOAT>("A", n, n);
    auto b = graph.make_variable<TF_FLOAT>("b", n, 1);
    auto x = graph.make_placeholder<TF_FLOAT>("x", n, 1);

    auto Ax = graph.make_matmul("Ax", A, x);
    auto Ax_plus_b = graph.make_addition("Ax_plus_b", Ax, b);

    Tensor2d A_init(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_init(i, j) = i * n + j + 2;
        }
    }
    TF_Operation* A_init_op = graph.make_variable_init("A_init", A, A_init);

    Tensor2d b_init(n, 1);
    for (int i = 0; i < n; ++i) {
        b_init(i, 0) = i + 3;
    }
    TF_Operation* b_init_op = graph.make_variable_init("b_init", b, b_init);

    graph.run_session(
            {},
            {},
            {},
            {A_init_op, b_init_op}
            );

    std::vector<Tensor2d> output_tensors = graph.run_session(
            {x},
            inputs,
            {Ax_plus_b},
            {}
            );

    cout << "Output: tensor\n";
    for (int i = 0; i < n; ++i) {
        cout << output_tensors[0](i, 0) << "\n";
    }
    return 0;
}
