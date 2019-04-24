module.exports = function (t) {
    var placeholderVis = {name: "tensor", path: "tensor.html"};
    var types = ["Tensor"]; // TODO add other types like Layers etc.
    return types.includes(t.constructor) ? [placeholderVis] : [];
};
