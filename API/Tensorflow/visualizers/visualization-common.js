(function () {
    document.addEventListener("keydown", function (e) {
        if (e.key == " " || e.key == "Escape")
            e.stopPropagation();
            e.preventDefault();
            window.frameElement.parentNode.dispatchEvent(new e.constructor(e.type, e))
    });

    document.addEventListener("keyup", function (e) {
        if ((!e.ctrlKey && e.key == " ") || e.key == "Escape")
            e.stopPropagation();
            e.preventDefault();
            window.frameElement.parentNode.dispatchEvent(new e.constructor(e.type, e))
    });
}());
