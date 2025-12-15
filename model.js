import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest';

const statusEl = () => document.getElementById('status');
const setStatus = (msg) => {
  console.log(msg);
  const el = statusEl();
  if (el) el.innerText = msg;
};

async function runDemo() {
  try {
    setStatus('Loading TensorFlow.js...');
    await tf.ready();
    setStatus('tf.ready() done — building model');

    const model = tf.sequential();
    model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 8,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu', padding: 'same' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // small demo: 3 classes

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    setStatus('Preparing synthetic data (quick test)...');
    const numSamples = 120;
    const xs = tf.randomNormal([numSamples, 28, 28, 1]);
    const labels = tf.randomUniform([numSamples], 0, 3, 'int32');
    const ys = tf.oneHot(labels, 3);

    setStatus('Training for 3 epochs (quick test)...');
    await model.fit(xs, ys, {
      epochs: 3,
      batchSize: 16,
      callbacks: {
        onEpochBegin: (epoch) => setStatus(`Epoch ${epoch + 1} start...`),
        onEpochEnd: (epoch, logs) => setStatus(`Epoch ${epoch + 1} done — loss=${logs.loss.toFixed(4)} acc=${(logs.acc || logs.accuracy).toFixed(4)}`)
      }
    });

    setStatus('Training finished — running a sample prediction');
    const sample = tf.randomNormal([1, 28, 28, 1]);
    const pred = model.predict(sample);
    pred.print();
    setStatus('Done — check console for details');

    // dispose tensors we created
    tf.dispose([xs, ys, labels, sample, pred]);
  } catch (err) {
    console.error(err);
    setStatus('Error: see console');
  }
}

runDemo();