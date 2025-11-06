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

export class StockGRUModel {
  constructor(config = {}) {
    this.sequenceLength = config.sequenceLength ?? 12;
    this.featureCount = config.featureCount ?? 20;
    this.stockCount = config.stockCount ?? 10;
    this.horizon = config.horizon ?? 3;
    this.learningRate = config.learningRate ?? 1e-3;

    this.modelPromise = this.#buildModel(config);
  }

  async #buildModel(config) {
    const tf = await getTF();
    const unitsFirst = config.unitsFirst ?? 128;
    const unitsSecond = config.unitsSecond ?? 64;
    const dropoutRate = config.dropoutRate ?? 0.2;

    const model = tf.sequential();
    model.add(
      tf.layers.gru({
        units: unitsFirst,
        returnSequences: true,
        inputShape: [this.sequenceLength, this.featureCount],
        kernelInitializer: 'glorotUniform',
      })
    );
    model.add(tf.layers.dropout({ rate: dropoutRate }));
    model.add(
      tf.layers.gru({
        units: unitsSecond,
        returnSequences: false,
        kernelInitializer: 'glorotUniform',
      })
    );
    model.add(tf.layers.dropout({ rate: dropoutRate }));
    model.add(
      tf.layers.dense({
        units: this.stockCount * this.horizon,
        activation: 'sigmoid',
        kernelInitializer: 'glorotUniform',
      })
    );

    const optimizer = tf.train.adam(this.learningRate);
    model.compile({
      optimizer,
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy'],
    });

    return model;
  }

  async ready() {
    if (!this.modelPromise) {
      throw new Error('Model was disposed.');
    }
    return this.modelPromise;
  }

  async train(X_train, y_train, options = {}) {
    const model = await this.ready();
    const trainOptions = {
      epochs: options.epochs ?? 30,
      batchSize: options.batchSize ?? 32,
      validationSplit: options.validationSplit ?? 0.1,
      shuffle: false,
      callbacks: this.#buildCallbacks(options.callbacks ?? {}),
    };

    return model.fit(X_train, y_train, trainOptions);
  }

  #buildCallbacks(callbackConfig) {
    const { onEpochEnd, onTrainBegin, onTrainEnd } = callbackConfig;
    return {
      onTrainBegin: async (logs) => {
        if (onTrainBegin) await onTrainBegin(logs ?? {});
      },
      onEpochEnd: async (epoch, logs) => {
        if (onEpochEnd) await onEpochEnd(epoch, logs ?? {});
        const tf = await getTF();
        await tf.nextFrame();
      },
      onTrainEnd: async (logs) => {
        if (onTrainEnd) await onTrainEnd(logs ?? {});
      },
    };
  }

  async predict(inputs) {
    const model = await this.ready();
    return model.predict(inputs);
  }

  async evaluateStockAccuracies(yTrue, yPred) {
    const tf = await getTF();
    return tf.tidy(() => {
      const predTensor = yPred instanceof tf.Tensor ? yPred : tf.tensor(yPred);
      const trueTensor = yTrue instanceof tf.Tensor ? yTrue : tf.tensor(yTrue);

      const binaryPred = predTensor.greaterEqual(0.5).toInt();
      const trueBinary = trueTensor.round().toInt();

      const reshapedPred = binaryPred.reshape([-1, this.stockCount, this.horizon]);
      const reshapedTrue = trueBinary.reshape([-1, this.stockCount, this.horizon]);

      const matchTensor = reshapedPred.equal(reshapedTrue);
      const accuracyPerStock = matchTensor.mean(2).mean(0);
      const data = accuracyPerStock.dataSync();
      return Array.from(data);
    });
  }

  async dispose() {
    if (!this.modelPromise) return;
    const model = await this.modelPromise.catch(() => null);
    if (model) {
      model.dispose();
    }
    this.modelPromise = null;
  }
}
