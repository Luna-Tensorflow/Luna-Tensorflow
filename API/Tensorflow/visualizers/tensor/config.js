module.exports = function (t) {
    var placeholderVis = {name: "tensor", path: "tensor.html"};
    var types = ["Tensor", "TFOutput"];
    if (types.includes(t.constructor)) {
        return [placeholderVis];
    }
    if (t.constructor == "List" && t.fields[0] && types.includes(t.fields[0])) {
        return [placeholderVis];
    }
    return [];
};
