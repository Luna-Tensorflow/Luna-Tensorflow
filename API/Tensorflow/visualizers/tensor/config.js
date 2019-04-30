module.exports = function (t) {
    var placeholderVis = {name: "tensor", path: "tensor.html"};
    var types = ["Tensor", "TFOutput"]; // TODO add other types like Layers etc.
    return types.includes(t.constructor) ? [placeholderVis] : [];
};
