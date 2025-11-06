import { DataLoader } from './data-loader.js';
import { StockGRUModel } from './gru.js';

const dom = {
  fileInput: document.getElementById('file-input'),
  trainButton: document.getElementById('train-btn'),
  evaluateButton: document.getElementById('evaluate-btn'),
  epochsInput: document.getElementById('epochs-input'),
  batchSizeInput: document.getElementById('batch-input'),
  statusLog: document.getElementById('status-log'),
  datasetSummary: document.getElementById('dataset-summary'),
  metricsSection: document.getElementById('metrics-section'),
  accuracyCanvas: document.getElementById('accuracy-chart'),
  timelineContainer: document.getElementById('timeline-container'),
};

const dataLoader = new DataLoader();
let dataset = null;
let model = null;
let accuracyChart = null;

function logStatus(message) {
  const timestamp = new Date().toLocaleTimeString();
  dom.statusLog.textContent += `[${timestamp}] ${message}\n`;
  dom.statusLog.scrollTop = dom.statusLog.scrollHeight;
}

function clearStatus() {
  dom.statusLog.textContent = '';
}

function resetVisualizations() {
  if (accuracyChart) {
    accuracyChart.destroy();
    accuracyChart = null;
  }
  dom.timelineContainer.innerHTML = '';
  dom.metricsSection.setAttribute('hidden', '');
}

function disposeDataset() {
  if (dataset) {
    dataset.dispose();
    dataset = null;
  }
}

function disableControls() {
  dom.trainButton.disabled = true;
  dom.evaluateButton.disabled = true;
}

function enableTrainingControls() {
  dom.trainButton.disabled = false;
  dom.evaluateButton.disabled = true;
}

function enableEvaluationControls() {
  dom.trainButton.disabled = false;
  dom.evaluateButton.disabled = false;
}

async function handleFileSelection(event) {
  const [file] = event.target.files;
  disposeDataset();
  resetVisualizations();
  clearStatus();

  if (!file) {
    disableControls();
    logStatus('No file selected.');
    return;
  }

  logStatus(`Loading file: ${file.name}`);
  try {
    const info = await dataLoader.loadFile(file);
    dom.datasetSummary.textContent = `Loaded ${info.symbols.length} symbols and ${info.dates.length} trading days.`;
    enableTrainingControls();
    logStatus('File loaded successfully. Ready to train.');
  } catch (error) {
    dom.datasetSummary.textContent = '';
    disableControls();
    logStatus(`Error: ${error.message}`);
    console.error(error);
  }
}

async function trainModel() {
  if (!dom.fileInput.files.length) {
    logStatus('Please upload a CSV file before training.');
    return;
  }

  disposeDataset();
  resetVisualizations();
  disableControls();

  logStatus('Preparing dataset...');
  try {
    dataset = await dataLoader.prepareDataset();
  } catch (error) {
    enableTrainingControls();
    logStatus(`Dataset error: ${error.message}`);
    console.error(error);
    return;
  }

  if (model) {
    await model.dispose();
    model = null;
  }

  model = new StockGRUModel({
    sequenceLength: dataset.sequenceLength,
    featureCount: dataset.featureCount,
    stockCount: dataset.stockSymbols.length,
    horizon: dataset.horizon,
  });

  try {
    await model.ready();
  } catch (error) {
    logStatus(`Model initialization error: ${error.message}`);
    console.error(error);
    enableTrainingControls();
    return;
  }

  const epochs = Number.parseInt(dom.epochsInput.value, 10) || 30;
  const batchSize = Number.parseInt(dom.batchSizeInput.value, 10) || 32;

  logStatus(`Training model for ${epochs} epochs (batch size ${batchSize})...`);

  try {
    await model.train(dataset.X_train, dataset.y_train, {
      epochs,
      batchSize,
      validationSplit: 0.1,
      callbacks: {
        onTrainBegin: () => logStatus('Training started.'),
        onEpochEnd: (epoch, logs) => {
          const message = `Epoch ${epoch + 1}/${epochs} — loss: ${logs.loss.toFixed(4)}, val_loss: ${
            logs.val_loss?.toFixed(4) ?? 'n/a'
          }, acc: ${logs.binaryAccuracy?.toFixed(4) ?? 'n/a'}`;
          logStatus(message);
        },
        onTrainEnd: () => logStatus('Training completed.'),
      },
    });
    enableEvaluationControls();
  } catch (error) {
    logStatus(`Training error: ${error.message}`);
    console.error(error);
    enableTrainingControls();
  }
}

async function evaluateModel() {
  if (!dataset || !model) {
    logStatus('Train the model before evaluation.');
    return;
  }

  disableControls();
  logStatus('Running evaluation on the test set...');

  let predictionTensor;
  try {
    predictionTensor = await model.predict(dataset.X_test);
    const stockAccuracies = await model.evaluateStockAccuracies(dataset.y_test, predictionTensor);
    const predictions = await predictionTensor.array();
    const groundTruth = await dataset.y_test.array();
    renderAccuracyChart(dataset.stockSymbols, stockAccuracies);
    renderTimelines(dataset, predictions, groundTruth);
    logStatus('Evaluation complete. Visualizations updated.');
    dom.metricsSection.removeAttribute('hidden');
  } catch (error) {
    logStatus(`Evaluation error: ${error.message}`);
    console.error(error);
  } finally {
    predictionTensor?.dispose();
    enableEvaluationControls();
  }
}

function renderAccuracyChart(symbols, accuracies) {
  const sorted = symbols.map((symbol, index) => ({ symbol, accuracy: accuracies[index] ?? 0 }));
  sorted.sort((a, b) => b.accuracy - a.accuracy);

  const labels = sorted.map((item) => item.symbol);
  const dataValues = sorted.map((item) => Number.isFinite(item.accuracy) ? item.accuracy : 0);

  if (accuracyChart) {
    accuracyChart.destroy();
  }

  accuracyChart = new Chart(dom.accuracyCanvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Binary Accuracy',
          data: dataValues,
          backgroundColor: dataValues.map((value) => (value >= 0.5 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(248, 113, 113, 0.8)')),
          borderRadius: 8,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: {
        x: {
          min: 0,
          max: 1,
          ticks: { callback: (value) => `${(value * 100).toFixed(0)}%` },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => `${(context.parsed.x * 100).toFixed(2)}% accuracy`,
          },
        },
      },
    },
  });
}

function renderTimelines(datasetInfo, predictions, groundTruth) {
  const { stockSymbols, horizon, testAnchorIndices, allDates } = datasetInfo;
  const stockCount = stockSymbols.length;
  const timelineData = Array.from({ length: stockCount }, () => []);

  predictions.forEach((samplePred, sampleIndex) => {
    const sampleTruth = groundTruth[sampleIndex];
    const anchorIndex = testAnchorIndices[sampleIndex];
    for (let stockIdx = 0; stockIdx < stockCount; stockIdx += 1) {
      for (let h = 0; h < horizon; h += 1) {
        const labelIndex = stockIdx * horizon + h;
        const predictedBinary = samplePred[labelIndex] >= 0.5 ? 1 : 0;
        const truthBinary = Math.round(sampleTruth[labelIndex]);
        const correct = predictedBinary === truthBinary;
        const futureDateIndex = anchorIndex + (h + 1);
        const futureDate = allDates[futureDateIndex];
        timelineData[stockIdx].push({
          horizon: h + 1,
          anchorDate: allDates[anchorIndex],
          futureDate,
          correct,
        });
      }
    }
  });

  dom.timelineContainer.innerHTML = '';

  stockSymbols.forEach((symbol, stockIdx) => {
    const section = document.createElement('section');
    section.className = 'timeline-section';

    const header = document.createElement('h3');
    header.textContent = `${symbol}`;
    section.appendChild(header);

    const row = document.createElement('div');
    row.className = 'timeline-row';

    timelineData[stockIdx].forEach((entry) => {
      const cell = document.createElement('div');
      cell.className = `timeline-cell ${entry.correct ? 'correct' : 'incorrect'}`;
      const label = `Anchor ${entry.anchorDate} → ${entry.futureDate} (D+${entry.horizon})`;
      cell.title = `${symbol}: ${label} — ${entry.correct ? 'Correct' : 'Incorrect'}`;
      row.appendChild(cell);
    });

    section.appendChild(row);
    dom.timelineContainer.appendChild(section);
  });
}

dom.fileInput.addEventListener('change', handleFileSelection);
dom.trainButton.addEventListener('click', () => {
  trainModel().catch((error) => {
    logStatus(`Unexpected training error: ${error.message}`);
    console.error(error);
  });
});
dom.evaluateButton.addEventListener('click', () => {
  evaluateModel().catch((error) => {
    logStatus(`Unexpected evaluation error: ${error.message}`);
    console.error(error);
  });
});

disableControls();
