#ifndef TFL_VARIABLE_H
#define TFL_VARIABLE_H


#include "../ops/Binder.h"

class Variable : public Binder {
public:
    std::string get_name();

    static std::shared_ptr<Output> make_variable(std::string& name,
        std::shared_ptr<Tensor> default_value);
    size_t hashcode() const override;
    std::string hash_log() const override;


private:
    Variable(std::string& name, std::shared_ptr<Tensor> default_value);
    void add_to_graph(GraphSession&) override;

    std::string name;
    std::shared_ptr<Tensor> default_value;
    std::weak_ptr<Output> my_output;
    size_t hash;
};


#endif //TFL_VARIABLE_H
