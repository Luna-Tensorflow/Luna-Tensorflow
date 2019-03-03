#ifndef TFL_STATE_H
#define TFL_STATE_H


#include <string>
#include <memory>
#include <map>
#include "../tensor/Tensor.h"

class State {
    std::map<std::string, std::shared_ptr<Tensor>> valuation;
public:
    // TODO not sure if return by val or ptr or ?
    std::shared_ptr<State> updated(const std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> &new_values) {
        // TODO
        auto s = std::make_shared<State>(*this);
        for (auto & p : new_values) {
            s->valuation[p.first] = p.second;
        }
        return s;
    }

    std::shared_ptr<State> updated(std::string name, std::shared_ptr<Tensor> value) {
        return updated({{name, value}});
    }

    static std::shared_ptr<State> make_empty() {
        return std::make_shared<State>();
    }

    // returns nullptr if values does not exist
    std::shared_ptr<Tensor> get(const std::string &name) {
        auto it = valuation.find(name);
        if (it == valuation.end()) {
            return nullptr;
        }
        return it->second;
    }

    std::vector<std::shared_ptr<Tensor>> get(const std::vector<std::string>& names)
    {
        std::vector<std::shared_ptr<Tensor>> values(names.size());
        std::transform(names.begin(), names.end(), values.begin(), [&](const std::string& name)
        {
            return get(name);
        });

        return values;
    }
};


#endif //TFL_STATE_H
