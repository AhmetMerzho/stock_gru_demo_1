class DataLoader {
    constructor() {
        this.stocksData = null;
        this.normalizedData = null;
        this.symbols = [];
        this.dates = [];
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
        this.testDates = [];
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    this.parseCSV(csv);
                    resolve(this.stocksData);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        
        const data = {};
        const symbols = new Set();
        const dates = new Set();

        // Parse all rows
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            if (values.length !== headers.length) continue;

            const row = {};
            headers.forEach((header, index) => {
                row[header.trim()] = values[index].trim();
            });

            const symbol = row.Symbol;
            const date = row.Date;
            
            symbols.add(symbol);
            dates.add(date);

            if (!data[symbol]) data[symbol] = {};
            data[symbol][date] = {
                Open: parseFloat(row.Open),
                Close: parseFloat(row.Close),
                High: parseFloat(row.High),
                Low: parseFloat(row.Low),
                Volume: parseFloat(row.Volume)
            };
        }

        this.symbols = Array.from(symbols).sort();
        this.dates = Array.from(dates).sort();
        this.stocksData = data;

        console.log(`Loaded ${this.symbols.length} stocks with ${this.dates.length} trading days`);
    }

    normalizeData() {
        if (!this.stocksData) throw new Error('No data loaded');
        
        this.normalizedData = {};
        const minMax = {};

        // Calculate min-max per stock for Open and Close
        this.symbols.forEach(symbol => {
            minMax[symbol] = {
                Open: { min: Infinity, max: -Infinity },
                Close: { min: Infinity, max: -Infinity }
            };

            this.dates.forEach(date => {
                if (this.stocksData[symbol][date]) {
                    const point = this.stocksData[symbol][date];
                    minMax[symbol].Open.min = Math.min(minMax[symbol].Open.min, point.Open);
                    minMax[symbol].Open.max = Math.max(minMax[symbol].Open.max, point.Open);
                    minMax[symbol].Close.min = Math.min(minMax[symbol].Close.min, point.Close);
                    minMax[symbol].Close.max = Math.max(minMax[symbol].Close.max, point.Close);
                }
            });
        });

        // Normalize data
        this.symbols.forEach(symbol => {
            this.normalizedData[symbol] = {};
            this.dates.forEach(date => {
                if (this.stocksData[symbol][date]) {
                    const point = this.stocksData[symbol][date];
                    this.normalizedData[symbol][date] = {
                        Open: (point.Open - minMax[symbol].Open.min) / 
                              (minMax[symbol].Open.max - minMax[symbol].Open.min),
                        Close: (point.Close - minMax[symbol].Close.min) / 
                               (minMax[symbol].Close.max - minMax[symbol].Close.min)
                    };
                }
            });
        });

        return this.normalizedData;
    }

    createSequences(sequenceLength = 12, predictionHorizon = 3) {
        if (!this.normalizedData) this.normalizeData();

        const sequences = [];
        const targets = [];
        const validDates = [];

        // Create aligned data matrix
        const alignedData = [];
        for (let i = sequenceLength; i < this.dates.length - predictionHorizon; i++) {
            const currentDate = this.dates[i];
            const sequenceData = [];
            let validSequence = true;

            // Get sequence for all symbols
            for (let j = sequenceLength - 1; j >= 0; j--) {
                const seqDate = this.dates[i - j];
                const timeStepData = [];

                this.symbols.forEach(symbol => {
                    if (this.normalizedData[symbol][seqDate]) {
                        timeStepData.push(
                            this.normalizedData[symbol][seqDate].Open,
                            this.normalizedData[symbol][seqDate].Close
                        );
                    } else {
                        validSequence = false;
                    }
                });

                if (validSequence) sequenceData.push(timeStepData);
            }

            // Create target labels
            if (validSequence) {
                const target = [];
                const baseClosePrices = [];

                // Get base close prices (current date)
                this.symbols.forEach(symbol => {
                    baseClosePrices.push(this.stocksData[symbol][currentDate].Close);
                });

                // Calculate binary labels for prediction horizon
                for (let offset = 1; offset <= predictionHorizon; offset++) {
                    const futureDate = this.dates[i + offset];
                    this.symbols.forEach((symbol, idx) => {
                        if (this.stocksData[symbol][futureDate]) {
                            const futureClose = this.stocksData[symbol][futureDate].Close;
                            target.push(futureClose > baseClosePrices[idx] ? 1 : 0);
                        } else {
                            validSequence = false;
                        }
                    });
                }

                if (validSequence) {
                    sequences.push(sequenceData);
                    targets.push(target);
                    validDates.push(currentDate);
                }
            }
        }

        // Split into train/test (80/20 chronological split)
        const splitIndex = Math.floor(sequences.length * 0.8);
        
        this.X_train = tf.tensor3d(sequences.slice(0, splitIndex));
        this.y_train = tf.tensor2d(targets.slice(0, splitIndex));
        this.X_test = tf.tensor3d(sequences.slice(splitIndex));
        this.y_test = tf.tensor2d(targets.slice(splitIndex));
        this.testDates = validDates.slice(splitIndex);

        console.log(`Created ${sequences.length} sequences`);
        console.log(`Training: ${this.X_train.shape[0]}, Test: ${this.X_test.shape[0]}`);
        
        return {
            X_train: this.X_train,
            y_train: this.y_train,
            X_test: this.X_test,
            y_test: this.y_test,
            symbols: this.symbols,
            testDates: this.testDates
        };
    }

    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}

export default DataLoader;
