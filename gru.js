export class StockGRUModel {
  constructor(config = {}) {
    this.sequenceLength = config.sequenceLength ?? 12;
    this.featureCount = config.featureCount ?? 20;
    this.stockCount = config.stockCount ?? 10;
    this.horizon = config.horizon ?? 3;
    this.learningRate = config.learningRate ?? 1e-3;

    this.model = this.#buildModel(config);
  }

  #buildModel(config) {
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

  async train(X_train, y_train, options = {}) {
    const trainOptions = {
      epochs: options.epochs ?? 30,
      batchSize: options.batchSize ?? 32,
      validationSplit: options.validationSplit ?? 0.1,
      shuffle: false,
      callbacks: this.#buildCallbacks(options.callbacks ?? {}),
    };

    return this.model.fit(X_train, y_train, trainOptions);
  }

  #buildCallbacks(callbackConfig) {
    const { onEpochEnd, onTrainBegin, onTrainEnd } = callbackConfig;
    return {
      onTrainBegin: async (logs) => {
        if (onTrainBegin) await onTrainBegin(logs ?? {});
      },
      onEpochEnd: async (epoch, logs) => {
        if (onEpochEnd) await onEpochEnd(epoch, logs ?? {});
        await tf.nextFrame();
      },
      onTrainEnd: async (logs) => {
        if (onTrainEnd) await onTrainEnd(logs ?? {});
      },
    };
  }

  predict(inputs) {
    return this.model.predict(inputs);
  }

  evaluateStockAccuracies(yTrue, yPred) {
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

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}
