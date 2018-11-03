#include <iostream>
#include <tensorflow/c/c_api.h>
#include <vector>
#include "Tensor2d.h"
#include "Graph.h"

using namespace std;

int main() {
    int n, m;
    cin >> n >> m;

    std::vector<Tensor2d> input_tensors;
    input_tensors.emplace_back(n, m);
    input_tensors.emplace_back(n, m);
    for (int k = 0; k < 2; ++k) {
        cout << "Tensor " << k << endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                float x;
                cin >> x;
                input_tensors[k](i, j) = x;
            }
        }
    }

    Graph graph;

    auto a = graph.make_variable<TF_FLOAT>("Var1", n, m);
    auto b = graph.make_variable<TF_FLOAT>("Var2", n, m);

    auto added = graph.make_addition("Add1", a, b);

    std::vector<Tensor2d> output_tensors = graph.run_session(
            {a, b},
            input_tensors,
            {added}
            );

    cout << "Output:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cout << output_tensors[0](i, j) << " ";
        }
        cout << "\n";
    }

    return 0;
}
