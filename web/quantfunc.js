import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "QuantFunc.TransformerFilter",

    nodeCreated(node) {
        if (node.comfyClass !== "QuantFuncModelAutoLoader") return;

        const seriesWidget = node.widgets.find(w => w.name === "model_series");
        const transformerWidget = node.widgets.find(w => w.name === "transformer");
        if (!seriesWidget || !transformerWidget) return;

        // Full option set as reported by the server (/object_info), captured
        // BEFORE any narrowing so every filter pass starts from the complete
        // list (never from a previously-narrowed one).
        const allOptions = [...transformerWidget.options.values];

        // preserveCurrent=true  -> keep the currently-selected value valid even
        //                          if it doesn't match the series (workflow load
        //                          restores the value; dropping it would make
        //                          ComfyUI flag an already-present model as
        //                          "missing" and wipe the user's choice).
        // preserveCurrent=false -> manual series change: drop a now-mismatched
        //                          selection back to "None".
        function filterTransformer(preserveCurrent) {
            const series = seriesWidget.value || "";
            // "QuantFunc/Klein-9B-Series" -> "Klein-9B-Series"
            const shortName = series.includes("/") ? series.split("/").pop() : series;

            const filtered = allOptions.filter(opt =>
                opt === "None" || opt.startsWith(shortName + "/")
            );
            if (filtered.length === 0) filtered.push("None");

            const current = transformerWidget.value;
            if (current && current !== "None" && !filtered.includes(current)) {
                if (preserveCurrent) {
                    filtered.push(current);          // keep restored value valid
                } else {
                    transformerWidget.value = "None"; // mismatched after a manual switch
                }
            }

            transformerWidget.options.values = filtered;
        }

        // Manual series change: re-filter and drop a mismatched selection.
        const origCallback = seriesWidget.callback;
        seriesWidget.callback = function (...args) {
            const r = origCallback ? origCallback.apply(this, args) : undefined;
            filterTransformer(false);
            return r;
        };

        // nodeCreated runs BEFORE configure() restores saved widget values, so
        // re-filter once the workflow has restored model_series + transformer.
        // Without this, options stay narrowed to the DEFAULT series and the
        // restored value (e.g. a Klein-9B transformer) is reported as a
        // "missing model" on every load.
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const r = origOnConfigure ? origOnConfigure.apply(this, args) : undefined;
            filterTransformer(true);
            return r;
        };

        // Initial pass (fresh node: value is the default "None").
        filterTransformer(true);
    }
});
