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
    Graph();

    TF_Operation* make_operation(const std::string &type, const std::string &name, const std::vector<TF_Output> &inputs, const std::vector<std::shared_ptr<Attr>> &attrs);

    TF_Output get_output(TF_Operation *operation, int index);

    TF_Operation* make_variable(const std::string &name, const std::vector<int64_t> &dims, TF_DataType dtype);

    TF_Operation* make_constant(const std::string &name, const Tensor &tensor);

    TF_Operation* make_assign(const std::string &name, TF_Output variable, TF_Output value);

    TF_Operation* make_variable_init(std::string name, TF_Output variable, const Tensor &tensor);

    TF_Operation* make_assign_sub(const std::string &name, TF_Output variable, TF_Output value);

    TF_Operation* make_placeholder(const std::string &name, const std::vector<int64_t> &dims, TF_DataType dtype);

    TF_Operation* make_addition(const std::string &name, TF_Output a, TF_Output b);

    TF_Operation* make_substraction(const std::string &name, TF_Output a, TF_Output b);

    TF_Operation* make_mul(const std::string &name, TF_Output a, TF_Output b);

    TF_Operation* make_square(const std::string &name, TF_Output a);

    std::vector<TF_Output> make_gradient(std::vector<TF_Output> ys, std::vector<TF_Output> xs);

    std::vector<Tensor> run_session(
            const std::vector<TF_Output> &inputs,
            const std::vector<Tensor> &input_values,
            const std::vector<TF_Output> &outputs,
            const std::vector<TF_Operation*> &operations
    );

    ~Graph();

private:
    TF_Graph *graph;
    TF_Session *session;
    TF_SessionOptions *options;
};

#endif //TF_EXAMPLE_GRAPH_H
