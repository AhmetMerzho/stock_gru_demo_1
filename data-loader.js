const TF_CDN_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.13.0/dist/tf.min.js';

const getTF = async () => {
  if (typeof globalThis === 'undefined') {
    throw new Error('TensorFlow.js requires a browser environment.');
  }

  if (typeof document === 'undefined') {
    throw new Error('TensorFlow.js requires a DOM to load its script.');
  }

  if (!globalThis.__tfReadyPromise) {
    globalThis.__tfReadyPromise = new Promise((resolve, reject) => {
      if (globalThis.tf && typeof globalThis.tf.ready === 'function') {
        globalThis.tf
          .ready()
          .then(() => resolve(globalThis.tf))
          .catch(reject);
        return;
      }

      const existing = Array.from(document.scripts).find((script) => script.src.includes('@tensorflow/tfjs'));
      const targetScript = existing ?? document.createElement('script');
      const cleanup = () => {
        targetScript.removeEventListener('load', handleLoad);
        targetScript.removeEventListener('error', handleError);
      };

      function handleLoad() {
        cleanup();
        targetScript.dataset.tfReady = 'true';
        if (globalThis.tf && typeof globalThis.tf.ready === 'function') {
          globalThis.tf
            .ready()
            .then(() => resolve(globalThis.tf))
            .catch(reject);
        } else {
          reject(new Error('TensorFlow.js script loaded but tf was not found on the global scope.'));
        }
      }

      function handleError(event) {
        cleanup();
        reject(new Error(`Unable to load TensorFlow.js script: ${event?.message ?? 'network error'}`));
      }

      if (!existing) {
        targetScript.defer = false;
        targetScript.async = false;
        targetScript.src = TF_CDN_URL;
        targetScript.crossOrigin = 'anonymous';
        targetScript.integrity = 'sha384-uE1YKKcf9zmXLh4FfLLRj1NfX3YAcaqpK+IylCvWbyevnED8GB5blzm0qrx+tr4V';
        targetScript.addEventListener('load', handleLoad, { once: true });
        targetScript.addEventListener('error', handleError, { once: true });
        targetScript.dataset.tfReady = 'loading';
        document.head.appendChild(targetScript);
      } else {
        existing.dataset.tfReady = existing.dataset.tfReady ?? 'loading';
        existing.addEventListener('load', handleLoad, { once: true });
        existing.addEventListener('error', handleError, { once: true });
        if (existing.dataset.tfReady === 'true' || globalThis.tf) {
          handleLoad();
        }
      }
    });
  }

  return globalThis.__tfReadyPromise;
};

export class DataLoader {
  constructor(options = {}) {
    this.sequenceLength = options.sequenceLength ?? 12;
    this.horizon = options.horizon ?? 3;
    this.featureCountPerStock = 2; // Open, Close
    this.splitRatio = options.splitRatio ?? 0.8;

    this.symbols = [];
    this.dates = [];
    this.rawDataBySymbol = new Map();
    this.normalizedDataBySymbol = new Map();
    this.anchorIndices = [];
  }

  async loadFile(file) {
    if (!(file instanceof File)) {
      throw new Error('A valid File object is required.');
    }

    const text = await file.text();
    this.#parseCSV(text);

    if (this.symbols.length === 0 || this.dates.length === 0) {
      throw new Error('No usable records were found in the CSV file.');
    }

    return {
      symbols: [...this.symbols],
      dates: [...this.dates],
      sequenceLength: this.sequenceLength,
      horizon: this.horizon,
    };
  }

  disposeDataset(dataset) {
    if (!dataset) return;
    dataset.X_train?.dispose();
    dataset.y_train?.dispose();
    dataset.X_test?.dispose();
    dataset.y_test?.dispose();
  }

  async prepareDataset() {
    if (this.symbols.length === 0 || this.dates.length === 0) {
      throw new Error('Load a CSV file before preparing the dataset.');
    }

    this.#normalize();
    const { inputs, labels, anchorIndices } = this.#buildSamples();

    if (inputs.length === 0) {
      throw new Error('Not enough data to create training samples.');
    }

    const split = this.#splitSamples(inputs, labels, anchorIndices);
    const featureSize = this.sequenceLength * this.symbols.length * this.featureCountPerStock;
    const labelSize = this.symbols.length * this.horizon;

    const tf = await getTF();

    const X_train = tf.tensor3d(
      this.#flatten(split.train.inputs, featureSize),
      [split.train.inputs.length, this.sequenceLength, this.symbols.length * this.featureCountPerStock]
    );
    const y_train = tf.tensor2d(
      this.#flatten(split.train.labels, labelSize),
      [split.train.labels.length, labelSize]
    );
    const X_test = tf.tensor3d(
      this.#flatten(split.test.inputs, featureSize),
      [split.test.inputs.length, this.sequenceLength, this.symbols.length * this.featureCountPerStock]
    );
    const y_test = tf.tensor2d(
      this.#flatten(split.test.labels, labelSize),
      [split.test.labels.length, labelSize]
    );

    return {
      X_train,
      y_train,
      X_test,
      y_test,
      stockSymbols: [...this.symbols],
      sequenceLength: this.sequenceLength,
      horizon: this.horizon,
      featureCount: this.symbols.length * this.featureCountPerStock,
      trainDates: split.train.dates,
      testDates: split.test.dates,
      testAnchorIndices: split.test.anchorIndices,
      allDates: [...this.dates],
      dispose: () => {
        X_train.dispose();
        y_train.dispose();
        X_test.dispose();
        y_test.dispose();
      },
    };
  }

  #parseCSV(text) {
    this.symbols = [];
    this.dates = [];
    this.rawDataBySymbol.clear();
    this.normalizedDataBySymbol.clear();

    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    if (lines.length <= 1) {
      throw new Error('The CSV file must include a header row and at least one record.');
    }

    const headers = lines[0].split(',').map((h) => h.trim());
    const headerIndex = (name) => headers.indexOf(name);
    const idxDate = headerIndex('Date');
    const idxSymbol = headerIndex('Symbol');
    const idxOpen = headerIndex('Open');
    const idxClose = headerIndex('Close');

    if ([idxDate, idxSymbol, idxOpen, idxClose].some((idx) => idx === -1)) {
      throw new Error('CSV file must contain Date, Symbol, Open, and Close columns.');
    }

    const tempDateSets = new Map();

    for (let i = 1; i < lines.length; i += 1) {
      const row = lines[i].split(',');
      if (row.length !== headers.length) {
        continue;
      }

      const date = row[idxDate].trim();
      const symbol = row[idxSymbol].trim();
      const open = parseFloat(row[idxOpen]);
      const close = parseFloat(row[idxClose]);

      if (!date || !symbol || Number.isNaN(open) || Number.isNaN(close)) {
        continue;
      }

      if (!this.rawDataBySymbol.has(symbol)) {
        this.rawDataBySymbol.set(symbol, new Map());
        tempDateSets.set(symbol, new Set());
      }

      this.rawDataBySymbol.get(symbol).set(date, { open, close });
      tempDateSets.get(symbol).add(date);
    }

    this.symbols = Array.from(this.rawDataBySymbol.keys()).sort();

    if (this.symbols.length === 0) {
      throw new Error('No stock symbols were detected in the CSV file.');
    }

    let commonDates = null;
    this.symbols.forEach((symbol) => {
      const dates = Array.from(tempDateSets.get(symbol)).sort();
      if (commonDates === null) {
        commonDates = new Set(dates);
      } else {
        commonDates = new Set(dates.filter((d) => commonDates.has(d)));
      }
    });

    if (!commonDates || commonDates.size === 0) {
      throw new Error('Stocks do not share a common set of dates.');
    }

    this.dates = Array.from(commonDates).sort();

    // Remove any rows that are not part of the common date set to keep alignment.
    this.symbols.forEach((symbol) => {
      const symbolData = this.rawDataBySymbol.get(symbol);
      const filtered = new Map();
      this.dates.forEach((date) => {
        const point = symbolData.get(date);
        if (point) {
          filtered.set(date, point);
        }
      });
      this.rawDataBySymbol.set(symbol, filtered);
    });
  }

  #normalize() {
    this.normalizedDataBySymbol.clear();

    this.symbols.forEach((symbol) => {
      const values = Array.from(this.rawDataBySymbol.get(symbol).values());
      let minOpen = Infinity;
      let maxOpen = -Infinity;
      let minClose = Infinity;
      let maxClose = -Infinity;

      values.forEach(({ open, close }) => {
        if (open < minOpen) minOpen = open;
        if (open > maxOpen) maxOpen = open;
        if (close < minClose) minClose = close;
        if (close > maxClose) maxClose = close;
      });

      const openRange = maxOpen - minOpen || 1;
      const closeRange = maxClose - minClose || 1;
      const normalized = new Map();

      this.dates.forEach((date) => {
        const point = this.rawDataBySymbol.get(symbol).get(date);
        const normOpen = (point.open - minOpen) / openRange;
        const normClose = (point.close - minClose) / closeRange;
        normalized.set(date, { open: normOpen, close: normClose });
      });

      this.normalizedDataBySymbol.set(symbol, normalized);
    });
  }

  #buildSamples() {
    const seqLen = this.sequenceLength;
    const horizon = this.horizon;
    const stockCount = this.symbols.length;
    const featuresPerTimestep = stockCount * this.featureCountPerStock;
    const inputs = [];
    const labels = [];
    const anchorIndices = [];

    for (let idx = seqLen - 1; idx < this.dates.length - horizon; idx += 1) {
      const featureVector = new Float32Array(seqLen * featuresPerTimestep);
      const labelVector = new Float32Array(stockCount * horizon);
      let featureOffset = 0;

      for (let step = idx - seqLen + 1; step <= idx; step += 1) {
        const stepDate = this.dates[step];
        for (let s = 0; s < stockCount; s += 1) {
          const symbol = this.symbols[s];
          const normPoint = this.normalizedDataBySymbol.get(symbol).get(stepDate);
          featureVector[featureOffset] = normPoint.open;
          featureVector[featureOffset + 1] = normPoint.close;
          featureOffset += this.featureCountPerStock;
        }
      }

      let labelOffset = 0;
      const baseDate = this.dates[idx];
      for (let s = 0; s < stockCount; s += 1) {
        const symbol = this.symbols[s];
        const baseClose = this.rawDataBySymbol.get(symbol).get(baseDate).close;
        for (let h = 1; h <= horizon; h += 1) {
          const futureDate = this.dates[idx + h];
          const futureClose = this.rawDataBySymbol.get(symbol).get(futureDate).close;
          labelVector[labelOffset] = futureClose > baseClose ? 1 : 0;
          labelOffset += 1;
        }
      }

      inputs.push(featureVector);
      labels.push(labelVector);
      anchorIndices.push(idx);
    }

    return { inputs, labels, anchorIndices };
  }

  #splitSamples(inputs, labels, anchorIndices) {
    const total = inputs.length;
    const trainCount = Math.min(total - 1, Math.max(1, Math.floor(total * this.splitRatio)));
    const testCount = total - trainCount;

    if (testCount <= 0) {
      throw new Error('Dataset split produced no test samples. Adjust the split ratio or provide more data.');
    }

    const split = {
      train: {
        inputs: inputs.slice(0, trainCount),
        labels: labels.slice(0, trainCount),
        dates: anchorIndices.slice(0, trainCount).map((i) => this.dates[i]),
        anchorIndices: anchorIndices.slice(0, trainCount),
      },
      test: {
        inputs: inputs.slice(trainCount),
        labels: labels.slice(trainCount),
        dates: anchorIndices.slice(trainCount).map((i) => this.dates[i]),
        anchorIndices: anchorIndices.slice(trainCount),
      },
    };

    return split;
  }

  #flatten(samples, sampleSize) {
    const buffer = new Float32Array(samples.length * sampleSize);
    samples.forEach((sample, index) => {
      buffer.set(sample, index * sampleSize);
    });
    return buffer;
  }
}
