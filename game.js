(function () {
    'use strict';

    // ========================================================================
    //  COLOR UTILITIES
    // ========================================================================
    // Maps value in [-1, 1] to teal (positive) → magenta (negative)
    function valueToColor(v, alpha) {
        const a = alpha !== undefined ? alpha : 1;
        const t = Math.max(-1, Math.min(1, v));
        if (t > 0) {
            // Teal (#21D6C6) blended with dark
            const s = t;
            const r = Math.floor(13 + s * (33 - 13));
            const g = Math.floor(0 + s * (214 - 0));
            const b = Math.floor(26 + s * (198 - 26));
            return `rgba(${r},${g},${b},${a})`;
        } else {
            // Magenta (#F000D2) blended with dark
            const s = -t;
            const r = Math.floor(13 + s * (240 - 13));
            const g = Math.floor(0 + s * 0);
            const b = Math.floor(26 + s * (210 - 26));
            return `rgba(${r},${g},${b},${a})`;
        }
    }

    function valueToHeatColor(v) {
        const t = Math.max(-1, Math.min(1, v));
        if (t > 0) {
            const s = t;
            return [Math.floor(13 + s * 20), Math.floor(s * 214), Math.floor(26 + s * 172)];
        } else {
            const s = -t;
            return [Math.floor(13 + s * 227), Math.floor(s * 0), Math.floor(26 + s * 184)];
        }
    }

    // ========================================================================
    //  DATASET GENERATION
    // ========================================================================
    function generateDataset(type, numPoints, noise) {
        const points = [];
        const noiseScale = noise / 100;

        switch (type) {
            case 'circle': {
                for (let i = 0; i < numPoints; i++) {
                    const angle = Math.random() * Math.PI * 2;
                    const r = Math.random();
                    const x = Math.cos(angle) * r;
                    const y = Math.sin(angle) * r;
                    const nx = x + (Math.random() - 0.5) * noiseScale;
                    const ny = y + (Math.random() - 0.5) * noiseScale;
                    const label = (r < 0.5) ? 1 : -1;
                    points.push({ x: nx, y: ny, label });
                }
                break;
            }
            case 'xor': {
                for (let i = 0; i < numPoints; i++) {
                    const x = Math.random() * 2 - 1;
                    const y = Math.random() * 2 - 1;
                    const nx = x + (Math.random() - 0.5) * noiseScale;
                    const ny = y + (Math.random() - 0.5) * noiseScale;
                    const label = (x * y >= 0) ? 1 : -1;
                    points.push({ x: nx, y: ny, label });
                }
                break;
            }
            case 'gaussian': {
                for (let i = 0; i < numPoints; i++) {
                    const label = (i % 2 === 0) ? 1 : -1;
                    const cx = label * 0.4;
                    const cy = 0;
                    const x = cx + randNormal() * 0.3 + (Math.random() - 0.5) * noiseScale;
                    const y = cy + randNormal() * 0.3 + (Math.random() - 0.5) * noiseScale;
                    points.push({ x: Math.max(-1, Math.min(1, x)), y: Math.max(-1, Math.min(1, y)), label });
                }
                break;
            }
            case 'spiral': {
                const perClass = Math.floor(numPoints / 2);
                for (let cls = 0; cls < 2; cls++) {
                    for (let i = 0; i < perClass; i++) {
                        const r = i / perClass;
                        const t = 1.75 * i / perClass * 2 * Math.PI + (cls * Math.PI);
                        const x = r * Math.cos(t) + (Math.random() - 0.5) * noiseScale * 0.5;
                        const y = r * Math.sin(t) + (Math.random() - 0.5) * noiseScale * 0.5;
                        points.push({
                            x: Math.max(-1, Math.min(1, x)),
                            y: Math.max(-1, Math.min(1, y)),
                            label: cls === 0 ? 1 : -1
                        });
                    }
                }
                break;
            }
        }
        return points;
    }

    function randNormal() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    // ========================================================================
    //  FEATURE EXTRACTION
    // ========================================================================
    const FEATURE_FUNCS = {
        x1: p => p.x,
        x2: p => p.y,
        x1sq: p => p.x * p.x,
        x2sq: p => p.y * p.y,
        x1x2: p => p.x * p.y,
        sinx1: p => Math.sin(p.x * Math.PI),
        sinx2: p => Math.sin(p.y * Math.PI)
    };

    function extractFeatures(point, activeFeatures) {
        return activeFeatures.map(f => FEATURE_FUNCS[f](point));
    }

    // ========================================================================
    //  NEURAL NETWORK
    // ========================================================================
    class NeuralNetwork {
        constructor(layerSizes, activation, regType, regRate) {
            this.layerSizes = layerSizes; // e.g. [2, 4, 4, 1]
            this.activation = activation;
            this.regType = regType;
            this.regRate = regRate;

            this.weights = [];  // weights[l][j][i] — weight from neuron i in layer l to neuron j in layer l+1
            this.biases = [];   // biases[l][j]

            // Initialize with Xavier/He initialization
            for (let l = 0; l < layerSizes.length - 1; l++) {
                const nIn = layerSizes[l];
                const nOut = layerSizes[l + 1];
                const scale = Math.sqrt(2.0 / nIn);
                const w = [];
                const b = [];
                for (let j = 0; j < nOut; j++) {
                    const wj = [];
                    for (let i = 0; i < nIn; i++) {
                        wj.push(randNormal() * scale);
                    }
                    w.push(wj);
                    b.push(0.01);
                }
                this.weights.push(w);
                this.biases.push(b);
            }
        }

        activate(x) {
            switch (this.activation) {
                case 'relu': return Math.max(0, x);
                case 'tanh': return Math.tanh(x);
                case 'sigmoid': return 1 / (1 + Math.exp(-x));
                case 'linear': return x;
                default: return Math.tanh(x);
            }
        }

        activateDerivative(output) {
            switch (this.activation) {
                case 'relu': return output > 0 ? 1 : 0;
                case 'tanh': return 1 - output * output;
                case 'sigmoid': return output * (1 - output);
                case 'linear': return 1;
                default: return 1 - output * output;
            }
        }

        forward(input) {
            const activations = [input];
            let current = input;
            for (let l = 0; l < this.weights.length; l++) {
                const next = [];
                const isOutput = (l === this.weights.length - 1);
                for (let j = 0; j < this.weights[l].length; j++) {
                    let sum = this.biases[l][j];
                    for (let i = 0; i < current.length; i++) {
                        sum += this.weights[l][j][i] * current[i];
                    }
                    // Output uses tanh for classification to stay in [-1,1]
                    next.push(isOutput ? Math.tanh(sum) : this.activate(sum));
                }
                activations.push(next);
                current = next;
            }
            return activations;
        }

        predict(input) {
            const acts = this.forward(input);
            return acts[acts.length - 1][0];
        }

        // Backprop + SGD step on a single sample
        trainStep(input, target, lr) {
            const activations = this.forward(input);
            const L = this.weights.length;
            const deltas = new Array(L);

            // Output layer delta
            const output = activations[L][0];
            const outputDeriv = 1 - output * output; // tanh derivative
            const error = output - target;
            deltas[L - 1] = [error * outputDeriv];

            // Hidden layer deltas
            for (let l = L - 2; l >= 0; l--) {
                deltas[l] = [];
                for (let i = 0; i < this.weights[l].length; i++) {
                    let sum = 0;
                    for (let j = 0; j < this.weights[l + 1].length; j++) {
                        sum += this.weights[l + 1][j][i] * deltas[l + 1][j];
                    }
                    const act = activations[l + 1][i];
                    deltas[l].push(sum * this.activateDerivative(act));
                }
            }

            // Update weights and biases
            for (let l = 0; l < L; l++) {
                for (let j = 0; j < this.weights[l].length; j++) {
                    for (let i = 0; i < this.weights[l][j].length; i++) {
                        let grad = deltas[l][j] * activations[l][i];
                        // Regularization
                        if (this.regType === 'l1') {
                            grad += this.regRate * Math.sign(this.weights[l][j][i]);
                        } else if (this.regType === 'l2') {
                            grad += this.regRate * this.weights[l][j][i];
                        }
                        this.weights[l][j][i] -= lr * grad;
                    }
                    this.biases[l][j] -= lr * deltas[l][j];
                }
            }

            return error * error;
        }

        // Train one epoch on a batch
        trainBatch(batch, lr) {
            let totalLoss = 0;
            for (const sample of batch) {
                totalLoss += this.trainStep(sample.input, sample.label, lr);
            }
            return totalLoss / batch.length;
        }

        // Compute loss on a dataset
        computeLoss(data) {
            let total = 0;
            for (const sample of data) {
                const pred = this.predict(sample.input);
                const err = pred - sample.label;
                total += err * err;
            }
            return total / data.length;
        }
    }

    // ========================================================================
    //  APPLICATION STATE
    // ========================================================================
    const State = {
        datasetType: 'circle',
        noise: 0,
        ratio: 50,
        batchSize: 10,
        learningRate: 0.003,
        activation: 'tanh',
        regType: 'none',
        regRate: 0,
        problemType: 'classification',

        hiddenLayers: [4, 2],  // neurons per hidden layer
        activeFeatures: ['x1', 'x2'],

        // Runtime
        network: null,
        trainData: [],
        testData: [],
        allData: [],
        epoch: 0,
        playing: false,
        trainLossHistory: [],
        testLossHistory: [],
        animId: null
    };

    // ========================================================================
    //  DATA MANAGEMENT
    // ========================================================================
    function regenerateData() {
        const total = 300;
        State.allData = generateDataset(State.datasetType, total, State.noise);
        splitData();
    }

    function splitData() {
        const all = [...State.allData];
        // Shuffle
        for (let i = all.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [all[i], all[j]] = [all[j], all[i]];
        }
        const splitIdx = Math.floor(all.length * State.ratio / 100);
        State.trainData = all.slice(0, splitIdx);
        State.testData = all.slice(splitIdx);
    }

    function prepareData(data) {
        return data.map(p => ({
            input: extractFeatures(p, State.activeFeatures),
            label: p.label,
            x: p.x,
            y: p.y
        }));
    }

    // ========================================================================
    //  NETWORK MANAGEMENT
    // ========================================================================
    function rebuildNetwork() {
        const inputSize = State.activeFeatures.length;
        if (inputSize === 0) return;
        const sizes = [inputSize, ...State.hiddenLayers, 1];
        State.network = new NeuralNetwork(sizes, State.activation, State.regType, State.regRate);
        State.epoch = 0;
        State.trainLossHistory = [];
        State.testLossHistory = [];
        updateEpochDisplay();
    }

    function resetAll() {
        stop();
        regenerateData();
        rebuildNetwork();
        renderAll();
    }

    // ========================================================================
    //  TRAINING LOOP
    // ========================================================================
    function trainOneEpoch() {
        if (!State.network || State.activeFeatures.length === 0) return;
        const trainPrepped = prepareData(State.trainData);
        const testPrepped = prepareData(State.testData);

        // Shuffle training data
        for (let i = trainPrepped.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [trainPrepped[i], trainPrepped[j]] = [trainPrepped[j], trainPrepped[i]];
        }

        // Mini-batch training
        for (let b = 0; b < trainPrepped.length; b += State.batchSize) {
            const batch = trainPrepped.slice(b, b + State.batchSize);
            State.network.trainBatch(batch, State.learningRate);
        }

        State.epoch++;
        const trainLoss = State.network.computeLoss(trainPrepped);
        const testLoss = State.network.computeLoss(testPrepped);
        State.trainLossHistory.push(trainLoss);
        State.testLossHistory.push(testLoss);

        updateEpochDisplay();
        document.getElementById('train-loss').textContent = trainLoss.toFixed(4);
        document.getElementById('test-loss').textContent = testLoss.toFixed(4);
    }

    function play() {
        State.playing = true;
        document.getElementById('btn-play').classList.add('playing');
        document.getElementById('icon-play').style.display = 'none';
        document.getElementById('icon-pause').style.display = '';
        tick();
    }

    function stop() {
        State.playing = false;
        document.getElementById('btn-play').classList.remove('playing');
        document.getElementById('icon-play').style.display = '';
        document.getElementById('icon-pause').style.display = 'none';
        if (State.animId) {
            cancelAnimationFrame(State.animId);
            State.animId = null;
        }
    }

    function tick() {
        if (!State.playing) return;
        // Run multiple epochs per frame for speed
        for (let i = 0; i < 3; i++) trainOneEpoch();
        renderAll();
        State.animId = requestAnimationFrame(tick);
    }

    function updateEpochDisplay() {
        document.getElementById('epoch-count').textContent =
            String(State.epoch).padStart(3, '0');
    }

    // ========================================================================
    //  RENDERING: DATASET THUMBNAILS
    // ========================================================================
    function renderDatasetThumbs() {
        const types = ['circle', 'xor', 'gaussian', 'spiral'];
        types.forEach(type => {
            const canvas = document.querySelector(`.dataset-thumb[data-set="${type}"]`);
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            ctx.fillStyle = '#0d001a';
            ctx.fillRect(0, 0, w, h);

            const pts = generateDataset(type, 80, 5);
            for (const p of pts) {
                const px = (p.x + 1) / 2 * w;
                const py = (1 - (p.y + 1) / 2) * h;
                ctx.beginPath();
                ctx.arc(px, py, 2, 0, Math.PI * 2);
                ctx.fillStyle = p.label > 0 ? '#21D6C6' : '#F000D2';
                ctx.fill();
            }
        });
    }

    // ========================================================================
    //  RENDERING: FEATURE THUMBNAILS
    // ========================================================================
    function renderFeatureThumbs() {
        const features = Object.keys(FEATURE_FUNCS);
        features.forEach(feat => {
            const item = document.querySelector(`.feature-item input[data-feat="${feat}"]`);
            if (!item) return;
            const canvas = item.parentElement.querySelector('canvas');
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const img = ctx.createImageData(w, h);
            const fn = FEATURE_FUNCS[feat];

            for (let py = 0; py < h; py++) {
                for (let px = 0; px < w; px++) {
                    const x = (px / w) * 2 - 1;
                    const y = 1 - (py / h) * 2;
                    const v = fn({ x, y });
                    const [r, g, b] = valueToHeatColor(v);
                    const idx = (py * w + px) * 4;
                    img.data[idx] = r;
                    img.data[idx + 1] = g;
                    img.data[idx + 2] = b;
                    img.data[idx + 3] = 255;
                }
            }
            ctx.putImageData(img, 0, 0);
        });
    }

    // ========================================================================
    //  RENDERING: OUTPUT HEATMAP
    // ========================================================================
    function renderOutput() {
        const canvas = document.getElementById('output-canvas');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const w = rect.width, h = rect.height;

        const discretize = document.getElementById('chk-discretize').checked;
        const showTest = document.getElementById('chk-test-data').checked;

        if (!State.network || State.activeFeatures.length === 0) {
            ctx.fillStyle = '#0d001a';
            ctx.fillRect(0, 0, w, h);
            ctx.fillStyle = '#AA66E0';
            ctx.font = '13px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Select at least one feature', w / 2, h / 2);
            return;
        }

        // Render heatmap
        const res = 60;
        const cellW = w / res;
        const cellH = h / res;
        for (let gy = 0; gy < res; gy++) {
            for (let gx = 0; gx < res; gx++) {
                const x = (gx / res) * 2 - 1;
                const y = 1 - (gy / res) * 2;
                const features = extractFeatures({ x, y }, State.activeFeatures);
                let v = State.network.predict(features);
                if (discretize) v = v > 0 ? 1 : -1;
                ctx.fillStyle = valueToColor(v, 0.6);
                ctx.fillRect(gx * cellW, gy * cellH, cellW + 1, cellH + 1);
            }
        }

        // Draw data points
        const drawPoints = (data, radius, alpha) => {
            for (const p of data) {
                const px = (p.x + 1) / 2 * w;
                const py = (1 - (p.y + 1) / 2) * h;
                ctx.beginPath();
                ctx.arc(px, py, radius, 0, Math.PI * 2);
                ctx.fillStyle = p.label > 0 ? `rgba(33,214,198,${alpha})` : `rgba(240,0,210,${alpha})`;
                ctx.fill();
                ctx.strokeStyle = `rgba(227,204,245,${alpha * 0.6})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        };

        drawPoints(State.trainData, 3, 0.9);
        if (showTest) drawPoints(State.testData, 2.5, 0.5);
    }

    // ========================================================================
    //  RENDERING: NETWORK VISUALIZATION
    // ========================================================================
    function renderNetwork() {
        const canvas = document.getElementById('network-canvas');
        const wrap = canvas.parentElement;
        const dpr = window.devicePixelRatio || 1;
        const rect = wrap.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const w = rect.width, h = rect.height;

        ctx.fillStyle = '#0d001a';
        ctx.fillRect(0, 0, w, h);

        if (!State.network) return;

        const layers = State.network.layerSizes;
        const numLayers = layers.length;
        const padX = 80;
        const padTop = 50;
        const padBottom = 20;
        const layerSpacing = (w - 2 * padX) / (numLayers - 1);
        const maxNeurons = Math.max(...layers);
        const neuronRadius = Math.min(18, (h - padTop - padBottom) / (maxNeurons * 3));

        // Compute neuron positions
        const positions = [];
        for (let l = 0; l < numLayers; l++) {
            const n = layers[l];
            const x = padX + l * layerSpacing;
            const totalHeight = (n - 1) * neuronRadius * 3;
            const startY = (h - padTop) / 2 + padTop / 2 - totalHeight / 2;
            const layerPos = [];
            for (let i = 0; i < n; i++) {
                layerPos.push({ x, y: startY + i * neuronRadius * 3 });
            }
            positions.push(layerPos);
        }

        // Draw connections
        for (let l = 0; l < State.network.weights.length; l++) {
            for (let j = 0; j < State.network.weights[l].length; j++) {
                for (let i = 0; i < State.network.weights[l][j].length; i++) {
                    const weight = State.network.weights[l][j][i];
                    const from = positions[l][i];
                    const to = positions[l + 1][j];
                    const absW = Math.abs(weight);
                    const lineWidth = Math.min(4, absW * 2);
                    if (lineWidth < 0.2) continue;
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.lineTo(to.x, to.y);
                    ctx.strokeStyle = valueToColor(weight, Math.min(0.8, absW * 0.5 + 0.1));
                    ctx.lineWidth = lineWidth;
                    ctx.stroke();
                }
            }
        }

        // Draw neurons
        for (let l = 0; l < numLayers; l++) {
            for (let i = 0; i < layers[l]; i++) {
                const pos = positions[l][i];
                const isInput = l === 0;
                const isOutput = l === numLayers - 1;

                // Neuron mini-heatmap
                if (!isInput) {
                    const bias = State.network.biases[l - 1][i];
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, neuronRadius, 0, Math.PI * 2);
                    ctx.fillStyle = valueToColor(bias, 0.3);
                    ctx.fill();
                }

                // Neuron border
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, neuronRadius, 0, Math.PI * 2);
                ctx.strokeStyle = isInput ? '#8E33D5' : (isOutput ? '#21D6C6' : '#7200CB');
                ctx.lineWidth = 2;
                ctx.stroke();

                // Neuron fill (dark center)
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, neuronRadius - 3, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(13,0,26,0.7)';
                ctx.fill();

                // Label
                if (isInput && State.activeFeatures[i]) {
                    ctx.fillStyle = '#C799EA';
                    ctx.font = '9px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    const labels = { x1: 'X₁', x2: 'X₂', x1sq: 'X₁²', x2sq: 'X₂²', x1x2: 'X₁X₂', sinx1: 'sin₁', sinx2: 'sin₂' };
                    ctx.fillText(labels[State.activeFeatures[i]] || State.activeFeatures[i], pos.x, pos.y);
                }
                if (isOutput) {
                    ctx.fillStyle = '#21D6C6';
                    ctx.font = '9px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('out', pos.x, pos.y);
                }
            }
        }

        // Layer labels
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (let l = 0; l < numLayers; l++) {
            const x = padX + l * layerSpacing;
            let label;
            if (l === 0) label = 'Input';
            else if (l === numLayers - 1) label = 'Output';
            else label = `Hidden ${l}`;
            ctx.fillStyle = '#AA66E0';
            ctx.fillText(label, x, 8);

            // +/- buttons for hidden layers
            if (l > 0 && l < numLayers - 1) {
                ctx.fillStyle = '#AA66E0';
                ctx.font = '10px monospace';
                ctx.fillText(`[${layers[l]}]`, x, 22);
            }
        }
    }

    // ========================================================================
    //  RENDERING: LOSS GRAPH
    // ========================================================================
    function renderLossGraph() {
        const canvas = document.getElementById('loss-canvas');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const w = rect.width, h = rect.height;

        ctx.fillStyle = '#0d001a';
        ctx.fillRect(0, 0, w, h);

        const trainH = State.trainLossHistory;
        const testH = State.testLossHistory;
        if (trainH.length < 2) {
            ctx.fillStyle = '#AA66E0';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Loss graph appears during training', w / 2, h / 2);
            return;
        }

        const allVals = [...trainH, ...testH];
        const maxVal = Math.max(...allVals, 0.01);
        const minVal = 0;
        const padL = 35, padR = 10, padT = 10, padB = 20;
        const gw = w - padL - padR;
        const gh = h - padT - padB;

        // Grid lines
        ctx.strokeStyle = '#2E0051';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = padT + gh * (1 - i / 4);
            ctx.beginPath();
            ctx.moveTo(padL, y);
            ctx.lineTo(w - padR, y);
            ctx.stroke();
            ctx.fillStyle = '#AA66E0';
            ctx.font = '8px monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText((maxVal * i / 4).toFixed(2), padL - 4, y);
        }

        // Draw lines
        const drawLine = (data, color) => {
            if (data.length < 2) return;
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            const step = gw / (data.length - 1);
            for (let i = 0; i < data.length; i++) {
                const x = padL + i * step;
                const y = padT + gh * (1 - (data[i] - minVal) / (maxVal - minVal));
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        };

        drawLine(trainH, '#21D6C6');
        drawLine(testH, '#F000D2');

        // Legend
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillStyle = '#21D6C6';
        ctx.fillText('Train', padL + 4, padT + 10);
        ctx.fillStyle = '#F000D2';
        ctx.fillText('Test', padL + 40, padT + 10);
    }

    // ========================================================================
    //  RENDERING: LAYER HEADERS WITH +/- BUTTONS
    // ========================================================================
    function renderLayerHeaders() {
        const container = document.getElementById('layer-headers');
        container.innerHTML = '';

        const numLayers = State.hiddenLayers.length;

        // Add-layer button at start
        const addLayerBtn = document.createElement('div');
        addLayerBtn.className = 'layer-header';
        addLayerBtn.innerHTML = `
            <span class="layer-title" style="font-size:0.65rem">HIDDEN LAYERS</span>
            <div class="layer-buttons">
                <button id="btn-add-layer" title="Add layer">+L</button>
                <button id="btn-rm-layer" title="Remove layer">-L</button>
            </div>
        `;
        container.appendChild(addLayerBtn);

        // Per-layer neuron controls
        for (let i = 0; i < numLayers; i++) {
            const div = document.createElement('div');
            div.className = 'layer-header';
            div.innerHTML = `
                <span class="layer-title">Layer ${i + 1}: ${State.hiddenLayers[i]}</span>
                <div class="layer-buttons">
                    <button data-action="add-neuron" data-layer="${i}" title="Add neuron">+</button>
                    <button data-action="rm-neuron" data-layer="${i}" title="Remove neuron">−</button>
                </div>
            `;
            container.appendChild(div);
        }

        // Event listeners
        document.getElementById('btn-add-layer').addEventListener('click', () => {
            if (State.hiddenLayers.length < 6) {
                State.hiddenLayers.push(2);
                rebuildNetwork();
                renderLayerHeaders();
                renderAll();
            }
        });
        document.getElementById('btn-rm-layer').addEventListener('click', () => {
            if (State.hiddenLayers.length > 1) {
                State.hiddenLayers.pop();
                rebuildNetwork();
                renderLayerHeaders();
                renderAll();
            }
        });
        container.querySelectorAll('[data-action="add-neuron"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const l = parseInt(btn.dataset.layer);
                if (State.hiddenLayers[l] < 8) {
                    State.hiddenLayers[l]++;
                    rebuildNetwork();
                    renderLayerHeaders();
                    renderAll();
                }
            });
        });
        container.querySelectorAll('[data-action="rm-neuron"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const l = parseInt(btn.dataset.layer);
                if (State.hiddenLayers[l] > 1) {
                    State.hiddenLayers[l]--;
                    rebuildNetwork();
                    renderLayerHeaders();
                    renderAll();
                }
            });
        });
    }

    // ========================================================================
    //  RENDER ALL
    // ========================================================================
    function renderAll() {
        renderOutput();
        renderNetwork();
        renderLossGraph();
    }

    // ========================================================================
    //  UI INITIALIZATION
    // ========================================================================
    function init() {
        // Render dataset thumbnails and feature thumbnails
        renderDatasetThumbs();
        renderFeatureThumbs();

        // Dataset picker
        document.getElementById('dataset-picker').addEventListener('click', (e) => {
            const thumb = e.target.closest('.dataset-thumb');
            if (!thumb) return;
            document.querySelectorAll('.dataset-thumb').forEach(t => t.classList.remove('active'));
            thumb.classList.add('active');
            State.datasetType = thumb.dataset.set;
            resetAll();
        });

        // Playback
        document.getElementById('btn-play').addEventListener('click', () => {
            if (State.playing) stop();
            else play();
        });
        document.getElementById('btn-reset').addEventListener('click', resetAll);
        document.getElementById('btn-step').addEventListener('click', () => {
            trainOneEpoch();
            renderAll();
        });

        // Top-bar controls
        document.getElementById('sel-lr').addEventListener('change', (e) => {
            State.learningRate = parseFloat(e.target.value);
        });
        document.getElementById('sel-activation').addEventListener('change', (e) => {
            State.activation = e.target.value;
            if (State.network) State.network.activation = e.target.value;
        });
        document.getElementById('sel-reg').addEventListener('change', (e) => {
            State.regType = e.target.value;
            if (State.network) State.network.regType = e.target.value;
        });
        document.getElementById('sel-reg-rate').addEventListener('change', (e) => {
            State.regRate = parseFloat(e.target.value);
            if (State.network) State.network.regRate = parseFloat(e.target.value);
        });
        document.getElementById('sel-problem').addEventListener('change', (e) => {
            State.problemType = e.target.value;
        });

        // Data controls
        document.getElementById('sl-ratio').addEventListener('input', (e) => {
            State.ratio = parseInt(e.target.value);
            document.getElementById('val-ratio').textContent = e.target.value + '%';
            splitData();
            renderAll();
        });
        document.getElementById('sl-noise').addEventListener('input', (e) => {
            State.noise = parseInt(e.target.value);
            document.getElementById('val-noise').textContent = e.target.value;
        });
        document.getElementById('sel-batch').addEventListener('change', (e) => {
            State.batchSize = parseInt(e.target.value);
        });
        document.getElementById('btn-regen').addEventListener('click', resetAll);

        // Feature toggles
        document.getElementById('features-grid').addEventListener('change', (e) => {
            const cb = e.target;
            if (!cb.dataset.feat) return;
            const item = cb.closest('.feature-item');
            if (cb.checked) item.classList.add('active');
            else item.classList.remove('active');

            State.activeFeatures = [];
            document.querySelectorAll('.feature-item input:checked').forEach(inp => {
                State.activeFeatures.push(inp.dataset.feat);
            });
            rebuildNetwork();
            renderAll();
        });

        // Output controls
        document.getElementById('chk-test-data').addEventListener('change', () => renderOutput());
        document.getElementById('chk-discretize').addEventListener('change', () => renderOutput());

        // Window resize
        window.addEventListener('resize', renderAll);

        // Layer headers
        renderLayerHeaders();

        // Initial state
        resetAll();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
