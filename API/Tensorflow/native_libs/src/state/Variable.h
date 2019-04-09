#ifndef TFL_VARIABLE_H
#define TFL_VARIABLE_H

#include <string>
#include <memory>

#include "../ops/Binder.h"
#include "../ops/Output.h"
#include "../tensor/Tensor.h"

class Variable : public Binder {
public:
    // result of this function lives only as long as the object itself
    const char* get_name() override;

    static std::shared_ptr<Output> make_variable(std::string& name,
        std::shared_ptr<Tensor> default_value);
    size_t hashcode() const override;
    std::string hash_log() const override;


    std::shared_ptr<Tensor> get_default_value() const;

private:
    Variable(std::string& name, std::shared_ptr<Tensor> default_value);
    void add_to_graph(GraphSession&) override;

    std::string name;
    std::shared_ptr<Tensor> default_value;
    std::weak_ptr<Output> my_output;
    size_t hash;
};


#endif //TFL_VARIABLE_H
