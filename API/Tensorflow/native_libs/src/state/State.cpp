#include "State.h"
#include "../helpers/LifeTimeManager.h"
#include "../state/Variable.h"

std::vector<std::shared_ptr<Tensor>> State::get_with_defaults(const std::vector<std::shared_ptr<Output>> vars) {
    std::vector<std::shared_ptr<Tensor>> values(vars.size());
    std::transform(vars.begin(), vars.end(), values.begin(), [&](const std::shared_ptr<Output>& out)
    {
        std::shared_ptr<Node> op = out->get_binder();
        std::shared_ptr<Variable> var = std::dynamic_pointer_cast<Variable>(op);
        if (var == nullptr) {
            throw std::runtime_error("Non-variable output provided to `State::get_with_defaults` as a variable");
        }

        std::string vname = var->get_name();
        auto tensor = get(vname);
        if (tensor) {
            return tensor;
        }
        // if our valuation doesn't have provided value, we return the variable's default value
        return var->get_default_value();
    });

    return values;
}
