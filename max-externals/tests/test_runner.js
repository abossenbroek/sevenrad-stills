// test_runner.js
// Automated test runner for SevenRad Max effects
// Reads manifest.json and executes all test cases

autowatch = 1;

// Global variables
var manifest = null;
var testImage = null;
var currentTestIndex = 0;
var allTests = [];

// Effect name mappings
var effectNames = {
    "noise": "sr.noise",
    "saturation": "sr.saturation",
    "chromatic": "sr.chromatic",
    "blur": "sr.blur",
    "blur_circular": "sr.blur.circular",
    "motion": "sr.motion",
    "saltpepper": "sr.saltpepper",
    "corduroy": "sr.corduroy",
    "bayer": "sr.bayer",
    "bandswap": "sr.bandswap",
    "downscale": "sr.downscale",
    "slcoff": "sr.slcoff",
    "corruption": "sr.corruption"
};

// Parameter type mappings (for mode values)
var modeMap = {
    "gaussian": 0,
    "row": 1,
    "column": 2
};

// Output functions
function log(msg) {
    post("TEST_SUITE: " + msg + "\n");
    outlet(0, msg);
}

function loadManifest() {
    var f = new File("reference/manifest.json", "read");
    if (f.isopen) {
        var content = "";
        while (f.position < f.eof) {
            content += f.readline();
        }
        f.close();

        try {
            manifest = JSON.parse(content);
            log("Manifest loaded successfully");
            return true;
        } catch(e) {
            log("ERROR: Failed to parse manifest.json - " + e);
            return false;
        }
    } else {
        log("ERROR: Could not open reference/manifest.json");
        return false;
    }
}

function buildTestList() {
    allTests = [];

    for (var effectName in manifest) {
        var testCases = manifest[effectName];

        // Skip empty arrays (like saltpepper, corruption)
        if (testCases.length === 0) {
            log("Skipping " + effectName + " (no test cases defined)");
            continue;
        }

        for (var i = 0; i < testCases.length; i++) {
            var testCase = testCases[i];
            allTests.push({
                effect: effectName,
                case_id: testCase.case_id,
                description: testCase.description,
                params_path: testCase.params_path,
                output_path: testCase.output_path
            });
        }
    }

    log("Built test list: " + allTests.length + " tests");
    return allTests.length > 0;
}

function loadTestParams(paramsPath) {
    var f = new File(paramsPath, "read");
    if (f.isopen) {
        var content = "";
        while (f.position < f.eof) {
            content += f.readline();
        }
        f.close();

        try {
            return JSON.parse(content);
        } catch(e) {
            log("ERROR: Failed to parse " + paramsPath + " - " + e);
            return null;
        }
    } else {
        log("ERROR: Could not open " + paramsPath);
        return null;
    }
}

function runTest(test) {
    log("Running: " + test.effect + "_" + test.case_id.toString().padStart(3, '0') + " - " + test.description);

    // Load parameters
    var params = loadTestParams(test.params_path);
    if (!params) {
        log("ERROR: Could not load parameters for " + test.effect);
        return false;
    }

    // Load test image
    var inputMatrix = new JitterMatrix("test_input");
    inputMatrix.read("input/test_image.png");

    // Get the actual effect name
    var effectGen = effectNames[test.effect];
    if (!effectGen) {
        log("ERROR: Unknown effect name: " + test.effect);
        return false;
    }

    // Apply effect based on type
    var outputMatrix = null;

    try {
        if (test.effect === "blur") {
            // Two-pass blur: horizontal then vertical
            outputMatrix = applyTwoPassBlur(inputMatrix, params.params.sigma);
        } else if (test.effect === "bayer") {
            // Two-pass bayer: mosaic then demosaic
            outputMatrix = applyBayer(inputMatrix, params.params.pattern);
        } else {
            // Single-pass effect
            outputMatrix = applySinglePassEffect(inputMatrix, effectGen, params.params);
        }

        if (outputMatrix) {
            // Save output to actual/ directory
            var outputFilename = "actual/" + test.effect + "_" + test.case_id.toString().padStart(3, '0') + ".png";
            outputMatrix.write(outputFilename);
            log("  Saved: " + outputFilename);

            // Send to preview
            outlet(0, "preview", outputMatrix.name);

            return true;
        } else {
            log("ERROR: Effect processing failed");
            return false;
        }
    } catch(e) {
        log("ERROR: Exception during test execution - " + e);
        return false;
    }
}

function applySinglePassEffect(inputMatrix, genName, params) {
    // Create jit.gl.pix object
    var pix = new JitterObject("jit.gl.pix");
    pix.gen = genName;

    // Set parameters
    for (var param in params) {
        var value = params[param];

        // Convert mode strings to integers
        if (param === "mode" && typeof value === "string") {
            value = modeMap[value] || 0;
        }

        // Convert pattern strings to integers for bayer
        if (param === "pattern" && typeof value === "string") {
            var patternMap = {"RGGB": 0, "BGGR": 1, "GRBG": 2, "GBRG": 3};
            value = patternMap[value] || 0;
        }

        // Set attribute
        pix.setattr(param, value);
    }

    // Process
    var outputMatrix = new JitterMatrix("test_output_" + Math.random());
    outputMatrix.frommatrix(inputMatrix);

    // Apply effect (this is simplified - in real Max you'd use jit.gl.texture)
    // For now, we'll use direct matrix operations which won't work for GPU shaders
    // This needs to be run in actual Max environment with proper GL context

    return outputMatrix;
}

function applyTwoPassBlur(inputMatrix, sigma) {
    // Horizontal pass
    var pix1 = new JitterObject("jit.gl.pix");
    pix1.gen = "sr.blur.h";
    pix1.setattr("sigma", sigma);

    var temp = new JitterMatrix("blur_temp_" + Math.random());
    temp.frommatrix(inputMatrix);

    // Vertical pass
    var pix2 = new JitterObject("jit.gl.pix");
    pix2.gen = "sr.blur.v";
    pix2.setattr("sigma", sigma);

    var outputMatrix = new JitterMatrix("blur_output_" + Math.random());
    outputMatrix.frommatrix(temp);

    return outputMatrix;
}

function applyBayer(inputMatrix, pattern) {
    // Convert pattern to integer
    var patternMap = {"RGGB": 0, "BGGR": 1, "GRBG": 2, "GBRG": 3};
    var patternInt = patternMap[pattern] || 0;

    // Mosaic pass
    var pix1 = new JitterObject("jit.gl.pix");
    pix1.gen = "sr.bayer.mosaic";
    pix1.setattr("pattern", patternInt);

    var temp = new JitterMatrix("bayer_temp_" + Math.random());
    temp.frommatrix(inputMatrix);

    // Demosaic pass
    var pix2 = new JitterObject("jit.gl.pix");
    pix2.gen = "sr.bayer.demosaic";
    pix2.setattr("pattern", patternInt);

    var outputMatrix = new JitterMatrix("bayer_output_" + Math.random());
    outputMatrix.frommatrix(temp);

    return outputMatrix;
}

function start() {
    log("=== Starting SevenRad Test Suite ===");

    // Load manifest
    if (!loadManifest()) {
        log("FAILED: Could not load manifest");
        return;
    }

    // Build test list
    if (!buildTestList()) {
        log("FAILED: No tests to run");
        return;
    }

    // Run all tests
    var passed = 0;
    var failed = 0;

    for (var i = 0; i < allTests.length; i++) {
        var test = allTests[i];
        if (runTest(test)) {
            passed++;
        } else {
            failed++;
        }
    }

    log("=== Test Suite Complete ===");
    log("Passed: " + passed);
    log("Failed: " + failed);
    log("Total:  " + allTests.length);
    log("Next step: Run compare_outputs.py to validate results");
}

// Entry point
function msg_int(v) {
    if (v === 1) {
        start();
    }
}

// Allow direct 'start' message
function anything() {
    if (messagename === "start") {
        start();
    }
}
