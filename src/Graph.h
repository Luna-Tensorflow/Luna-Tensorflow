//
// Created by Radek on 03.11.2018.
//

#ifndef TF_EXAMPLE_OPS_H
#define TF_EXAMPLE_OPS_H

#include <tensorflow/c/c_api.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "Tensor2d.h"


class TFException : public std::exception {
private:
    TF_Status* status;
public:
    // exception takes ownership of the status
    TFException(TF_Status* status) : status(status) {}

    ~TFException() override {
        TF_DeleteStatus(status);
    }
    const char* what() const noexcept override {
        return TF_Message(status);
    }
};

void delete_or_throw(TF_Status* status) {
    if (TF_GetCode(status) == TF_OK) {
        TF_DeleteStatus(status);
    } else {
        throw TFException(status);
    }
}

class Graph {
public:
    Graph() {
        graph = TF_NewGraph();
    }

    template<TF_DataType dtype> TF_Output make_variable(std::string name, int n, int m) {
        TF_OperationDescription *desc = TF_NewOperation(graph, "Variable", name.c_str());

        int64_t dims[] = {n,m};

        TF_SetAttrShape(desc, "shape", dims, 2);
        TF_SetAttrType(desc, "dtype", dtype);
        TF_Status* status = TF_NewStatus();
        TF_Operation* operation = TF_FinishOperation(desc, status);

        delete_or_throw(status);

        TF_Output output = {
                .oper = operation,
                .index = 0
        };
        return output;
    }

    TF_Output make_addition(std::string name, TF_Output a, TF_Output b) {
        TF_OperationDescription* desc = TF_NewOperation(graph, "Add", name.c_str());

        TF_AddInput(desc, a);
        TF_AddInput(desc, b);

        TF_Status* status = TF_NewStatus();
        TF_Operation* operation = TF_FinishOperation(desc, status);
        delete_or_throw(status);

        TF_Output output = {
                .oper = operation,
                .index = 0
        };
        return output;
    }

    std::vector<Tensor2d> run_session(
            const std::vector<TF_Output> &inputs,
            const std::vector<Tensor2d> &input_values,
            const std::vector<TF_Output> &outputs
    ) {

        if (inputs.size() != input_values.size()) throw std::invalid_argument("Input vectors must have same length");

        TF_SessionOptions* options = TF_NewSessionOptions();
        TF_Status *status = TF_NewStatus();
        TF_Session* session = TF_NewSession(graph, options, status);

        delete_or_throw(status);

        std::vector<TF_Tensor*> input_t;
        for (const Tensor2d& t : input_values) {
            input_t.push_back(t.get_underlying());
        }

        std::vector<TF_Tensor*> output_t(outputs.size());

        status = TF_NewStatus();
        TF_SessionRun(session,
                      nullptr,
                      inputs.data(), input_t.data(), inputs.size(), // we provide our input tensors by setting the outputs of the variable nodes
                      outputs.data(), output_t.data(), outputs.size(), // we specify which outputs we want to learn
                      nullptr, 0, // we could provide some nodes that we want to execute, but don't need their output (it makes sense because of nodes like "Assign")
                      nullptr,
                      status);
        void* p = TF_TensorData(output_t[0]);
        delete_or_throw(status);

        status = TF_NewStatus();
        TF_CloseSession(session, status);
        delete_or_throw(status);

        TF_DeleteSessionOptions(options);
        status = TF_NewStatus();
        TF_DeleteSession(session, status);
        delete_or_throw(status);

        std::vector<Tensor2d> output_values;
        std::transform(output_t.begin(), output_t.end(), std::back_inserter(output_values), [](TF_Tensor* tensor){return Tensor2d(tensor);});
        return output_values;
    }

    ~Graph() {
        TF_DeleteGraph(graph);
        graph = nullptr;
    }

private:
    TF_Graph *graph;
};

#endif //TF_EXAMPLE_OPS_H
