(function () {
  var entityMap = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
    '/': '&#x2F;',
    '`': '&#x60;',
    '=': '&#x3D;'
  };

  var escapeHtml = function (string) {
    return string.replace(/[&<>"'`=\/]/g, function (s) {
      return entityMap[s];
    });
  };

    function shape(json) {
        if (!Array.isArray(json)) {
            return [];
        }

        if (json.length == 0) {
            return [0];
        }

        return [json.length].concat(shape(json[0]));
    }

    function rank(json) {
        if (!Array.isArray(json)) {
            return 0;
        }
        return 1 + rank(json[0]);
    }

    function makeVis(vis) {
        document.body.innerHTML = //'<div class="container small">small vis TODO</div>' +
            '<div class="container big">' + vis + '</div>';
    }

    function findMinMax(data) {
        if (Array.isArray(data)) {
            var min = Infinity;
            var max = -Infinity;
            for (var x = 0; x < data.length; ++x) {
                var stat = findMinMax(data[x]);
                min = Math.min(min, stat.min);
                max = Math.max(max, stat.max);
            }
            return {min: min, max: max};
        } else {
            return {min: data, max: data};
        }
    }

    function blackAndWhiteMinMaxColoring(min, max) {
        function rescale(x) {
            return (x - min) / (max - min);
        }

        return function(v) {
            v = Math.round(rescale(v) * 255);
            if (v > 255) {
                v = 255;
            }
            if (v < 0) {
                v = 0;
            }
            var hex = v.toString(16);
            if (hex.length < 2) {
                hex = "0" + hex;
            }
            return "#" + hex + hex + hex;
        };
    }

    function trunc(x) {
        if (x == 0) {
            return "0";
        }
        if (x > 0.01 && x < 10000) {
            return "" + Math.round(100 * x) / 100;
        } else {
            return x.toExponential(3);
        }
    }

    function renderToCanvas(canvas, data, valueToColor, addText) {
        var sh = shape(data);
        var w = sh[0];
        var h = sh[1];

        var ctx = canvas.getContext("2d");
        var canvasSize = 200;
        ctx.fillStyle = '#BBCC00';
        var side = (canvasSize - 4) / Math.max(w,h);
        ctx.fillRect(0,0,4 + h * side,4 + w * side);
        for (var x = 0; x < w; ++x) {
            for (var y = 0; y < h; ++y) {
                var v = data[x][y];
                ctx.fillStyle = valueToColor(v);
                ctx.fillRect(y * side + 2, x * side + 2, side, side);
                if (addText && side > 30) {
                    // for small pictures (so pixels are big), add text values
                    ctx.fillStyle = '#00AA00';
                    ctx.fillText(trunc(v), y * side + 10, x * side + 15);
                }
            }
        }
    }
    function describe(sh, stat) {
        return 'Shape: [' + sh + '], values range from ' + trunc(stat.min) + ' to ' + trunc(stat.max);
    }
    function render2d(data, coloring) {
        var sh = shape(data);
        var w = sh[0];
        var h = sh[1];
        if (w == 0 || h == 0) {
            makeVis("Width or height is 0, shape: " + sh);
            return;
        }

        var stat = findMinMax(data);
        var valueToColor = coloring(stat.min, stat.max);

        document.body.innerHTML = '<div class="container big">' + describe(sh, stat) + '<canvas id="viscanvas" width="400px" height="400px"></div>';
        var canvas = document.getElementById("viscanvas");
        renderToCanvas(canvas, data, valueToColor, true);
    }

    function render3dStacked(data, coloring) {
        var sh = shape(data);
        var html = "";
        // TODO we want to stack along last not first dimension, this requires a reshape
        for (var i = 0; i < data.length; ++i) {
            html += '<div><canvas id="viscanvas' + i + '" width="200px" height="200px"></div><br>';
        }
        var stat = findMinMax(data);
        document.body.innerHTML = '<div class="container big">' + describe(sh, stat) + html + '</div>';

        var valueToColor = coloring(stat.min, stat.max);
        for (var i = 0; i < data.length; ++i) {
            var canvas = document.getElementById("viscanvas" + i);
            renderToCanvas(canvas, data[i], valueToColor, true);
        }
    }

    function renderRGB(data) {
        var stat = findMinMax(data);
        var scale = 1;
        if (stat.max > 1 && stat.max <= 255) {
            scale = 255;
        }
        function rgbColor(v) {
            function toHex(x) {
                var hex = Math.round(255 * x / scale).toString(16);
                if (hex.length < 2) {
                    hex = "0" + hex;
                }
                return hex;
            }
            return "#" + toHex(v[0]) + toHex(v[1]) + toHex(v[2]);
        }
        document.body.innerHTML = '<div class="container big">' + '<canvas id="viscanvas" width="400px" height="400px"></div>';
        var canvas = document.getElementById("viscanvas");
        renderToCanvas(canvas, data, rgbColor);
    }

    var render = function (data) {
        var vis = JSON.stringify(data);

        var r = rank(data);
        if (r > 3) {
            return makeVis("Visualization of tensors of ranks higher than 3 are not supported right now");
        }
        if (r == 3) {
            // TODO 3d visualization - show multiple 2d images side by side
            if (shape(data)[2] == 3) {
                renderRGB(data);
            } else {
                render3dStacked(data, blackAndWhiteMinMaxColoring);
            }
        }
        if (r == 2) {
            return render2d(data, blackAndWhiteMinMaxColoring);
        }
        if (r == 1) {
            return render2d([data], blackAndWhiteMinMaxColoring);
        }
        if (r == 0) {
            return makeVis(data); // scalar value, no sense to show image
        }
        return "Error";
    };

  window.addEventListener("message", function (evt) {
    d = JSON.parse(evt.data.data);
    render(d);
  });
}());
