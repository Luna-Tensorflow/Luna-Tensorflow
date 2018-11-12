//
// Created by Radek on 03.11.2018.
//

#ifndef TF_EXAMPLE_GRAPH_H
#define TF_EXAMPLE_GRAPH_H

#include <tensorflow/c/c_api.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <memory>

#include "Tensor.h"
#include "TFException.h"
#include "Attr.h"

class Graph {
public:
    Graph() {
        graph = TF_NewGraph();
        options = TF_NewSessionOptions();
        TF_Status *status = TF_NewStatus();
        session = TF_NewSession(graph, options, status);
        delete_or_throw(status);
    }

    TF_Operation* make_operation(const std::string &type, const std::string &name, const std::vector<TF_Output> &inputs, const std::vector<std::shared_ptr<Attr>> &attrs) {
        TF_OperationDescription *desc = TF_NewOperation(graph, type.c_str(), name.c_str());

        for (auto &input : inputs) {
            TF_AddInput(desc, input);
        }

        for (auto &attr: attrs) {
            attr->set(desc);
        }

        TF_Status *status = TF_NewStatus();

        TF_Operation *operation = TF_FinishOperation(desc, status);

        delete_or_throw(status);

        return operation;
    }

    TF_Output get_output(TF_Operation *operation, int index) {
        return {
            .oper = operation,
            .index = index
        };
    }

    TF_Operation* make_variable(const std::string &name, const std::vector<int64_t> &dims, TF_DataType dtype) {
        return make_operation("Variable", name, {},
                {std::make_shared<AttrShape>("shape", dims), std::make_shared<AttrType>("dtype", dtype)});
    }

    TF_Operation* make_constant(const std::string &name, const Tensor &tensor) {
        return make_operation("Const", name, {},
                {std::make_shared<AttrType>("dtype", TF_TensorType(tensor.get_underlying())), std::make_shared<AttrTensor>("value", tensor)});
    }

    TF_Operation* make_assign(const std::string &name, TF_Output variable, TF_Output value) {
        return make_operation("Assign", name,
                {variable, value}, {});
    }

    TF_Operation* make_variable_init(std::string name, TF_Output variable, const Tensor &tensor) {
        TF_Output value = get_output(make_constant(name + "_val", tensor), 0);
        return make_assign(name + "_assign", variable, value);
    }

    TF_Operation* make_assign_sub(const std::string &name, TF_Output variable, TF_Output value) {
        return make_operation("AssignSub", name,
                              {variable, value}, {});
    }

    TF_Operation* make_placeholder(const std::string &name, const std::vector<int64_t> &dims, TF_DataType dtype) {
        return make_operation("Placeholder", name, {},
                {std::make_shared<AttrShape>("shape", dims), std::make_shared<AttrType>("dtype", dtype)});
    }

    TF_Operation* make_addition(const std::string &name, TF_Output a, TF_Output b) {
        return make_operation("Add", name,
                {a, b}, {});
    }

    TF_Operation* make_substraction(const std::string &name, TF_Output a, TF_Output b) {
        return make_operation("Sub", name,
                              {a, b}, {});
    }

    TF_Operation* make_mul(const std::string &name, TF_Output a, TF_Output b) {
        return make_operation("Mul", name,
                              {a, b}, {});
    }

    TF_Operation* make_square(const std::string &name, TF_Output a) {
        return make_operation("Square", name,
                              {a}, {});
    }

    std::vector<TF_Output> make_gradient(std::vector<TF_Output> ys, std::vector<TF_Output> xs) {
        std::vector<TF_Output> dys(xs.size());

        TF_Status *status = TF_NewStatus();
        TF_AddGradients(graph, ys.data(), ys.size(), xs.data(), xs.size(), nullptr, status, dys.data());
        delete_or_throw(status);

        return dys;
    }

    std::vector<Tensor> run_session(
            const std::vector<TF_Output> &inputs,
            const std::vector<Tensor> &input_values,
            const std::vector<TF_Output> &outputs,
            const std::vector<TF_Operation*> &operations
    ) {

        if (inputs.size() != input_values.size()) throw std::invalid_argument("Input vectors must have same length");

        std::vector<TF_Tensor*> input_t;
        for (const Tensor& t : input_values) {
            input_t.push_back(t.get_underlying());
        }

        std::vector<TF_Tensor*> output_t(outputs.size());

        TF_Status* status = TF_NewStatus();
        TF_SessionRun(session,
                      nullptr,
                      inputs.data(), input_t.data(), inputs.size(), // we provide our input tensors by setting the outputs of the variable nodes
                      outputs.data(), output_t.data(), outputs.size(), // we specify which outputs we want to learn
                      operations.data(), operations.size(), // we can provide some nodes that we want to execute, but don't need their output (it makes sense because of nodes like "Assign")
                      nullptr,
                      status);
        delete_or_throw(status);

        std::vector<Tensor> output_values;
        std::transform(output_t.begin(), output_t.end(), std::back_inserter(output_values), [](TF_Tensor* tensor){return Tensor(tensor);});
        return output_values;
    }

    ~Graph() {
        TF_DeleteGraph(graph);
        graph = nullptr;
        TF_Status* status = TF_NewStatus();
        TF_CloseSession(session, status);
        delete_or_throw(status);

        TF_DeleteSessionOptions(options);
        options = nullptr;
        status = TF_NewStatus();
        TF_DeleteSession(session, status);
        delete_or_throw(status);
        session = nullptr;
    }

private:
    TF_Graph *graph;
    TF_Session *session;
    TF_SessionOptions *options;
};

#endif //TF_EXAMPLE_GRAPH_H
