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

    function makeVis(vis) {
        document.body.innerHTML = //'<div class="container small">small vis TODO</div>' +
            '<div class="container big">' + vis + '</div>';
    }

    function render2d(data) {
        var sh = shape(data);
        var w = sh[0];
        var h = sh[1];
        if (w == 0 || h == 0) {
            return makeVis("Width or height is 0, shape: " + sh);
        }
        function valueToColor(val) {
            var v = Math.round(val * 255);
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
        }
        function trunc(x) {
            return Math.round(100 * x) / 100;
        }
        var canvasSize = 200;
        var side = canvasSize / Math.max(w,h);

        document.body.innerHTML = '<div class="container big"><div id="debug"></div>debug info: ' + sh + ', ' + side + '<canvas id="viscanvas" width="400px" height="400px"></div>';
        var canvas = document.getElementById("viscanvas");
        var ctx = canvas.getContext("2d");
        var dbg = "";
        for (var x = 0; x < w; ++x) {
            for (var y = 0; y < h; ++y) {
                var v = data[x][y];
                ctx.fillStyle = valueToColor(v);
                ctx.fillRect(x * side, y * side, side, side);
                if (side > 40) {
                    // for small pictures (so pixels are big), add text values
                    ctx.fillStyle = '#00AA00';
                    ctx.fillText("" + trunc(v), x * side + 10, y * side + 10);
                }
            }
        }
        document.getElementById("debug").innerHTML = dbg;
        return null;
    }

    var render = function (data) {
        var vis = JSON.stringify(data);

        var sh = shape(data);
        if (sh.length > 3) {
            return makeVis("Visualization of tensors of ranks higher than 3 are not supported right now");
        }
        if (sh.length == 3) {
            // TODO 3d visualization - show multiple 2d images side by side
        }
        if (sh.length == 2) {
            return render2d(data);
        }
        if (sh.length == 1) {
            return render2d([data]);
        }
        if (sh.length == 0) {
            return makeVis(data); // scalar value, no sense to show image
        }
        return "Error";
    };

  window.addEventListener("message", function (evt) {
    d = JSON.parse(evt.data.data);
    render(d);
  });
}());
