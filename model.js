import React, { useState, useRef, useEffect } from 'react';
import * as tf from 'tensorflow';

export default function FaceCNN() {
  const [model, setModel] = useState(null);
  const [training, setTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);
  const trainInputRef = useRef(null);

  // Build the CNN model
  const buildModel = () => {
    const model = tf.sequential({
      layers: [
        // Conv block 1
        tf.layers.conv2d({
          inputShape: [128, 128, 3],
          filters: 32,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same'
        }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        
        // Conv block 2
        tf.layers.conv2d({
          filters: 64,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same'
        }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        
        // Conv block 3
        tf.layers.conv2d({
          filters: 128,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same'
        }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        
        // Dense layers
        tf.layers.flatten(),
        tf.layers.dropout({ rate: 0.5 }),
        tf.layers.dense({ units: 256, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.3 }),
        
        // Output: age (1), gender (2), ethnicity (5) = 8 total
        tf.layers.dense({ units: 8 })
      ]
    });

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mae']
    });

    return model;
  };

  // Parse UTK filename: age_gender_race_date.jpg
  const parseFilename = (filename) => {
    const parts = filename.split('_');
    if (parts.length < 3) return null;
    
    const age = parseInt(parts[0]);
    const gender = parseInt(parts[1]); // 0=male, 1=female
    const race = parseInt(parts[2]); // 0-4: White, Black, Asian, Indian, Others
    
    if (isNaN(age) || isNaN(gender) || isNaN(race)) return null;
    
    return { age, gender, race };
  };

  // Preprocess image
  const preprocessImage = async (file) => {
    return new Promise((resolve) => {
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      img.onload = () => {
        canvas.width = 128;
        canvas.height = 128;
        ctx.drawImage(img, 0, 0, 128, 128);
        
        const tensor = tf.browser.fromPixels(canvas)
          .toFloat()
          .div(255.0)
          .expandDims(0);
        
        resolve(tensor);
      };
      
      img.src = URL.createObjectURL(file);
    });
  };

  // Create target tensor from labels
  const createTarget = (labels) => {
    // Normalize age to 0-1 range (assuming max age 100)
    const ageNorm = labels.age / 100;
    
    // One-hot encode gender (2 values)
    const gender = labels.gender === 0 ? [1, 0] : [0, 1];
    
    // One-hot encode race (5 values)
    const race = [0, 0, 0, 0, 0];
    race[labels.race] = 1;
    
    return tf.tensor2d([[ageNorm, ...gender, ...race]]);
  };

  // Train the model
  const trainModel = async () => {
    const files = trainInputRef.current.files;
    if (files.length === 0) {
      alert('Please select training images from the UTK Face dataset');
      return;
    }

    setTraining(true);
    setEpoch(0);
    
    const newModel = buildModel();
    
    // Prepare training data
    const trainData = [];
    const trainLabels = [];
    
    for (let i = 0; i < Math.min(files.length, 100); i++) {
      const file = files[i];
      const labels = parseFilename(file.name);
      
      if (!labels) continue;
      
      const imgTensor = await preprocessImage(file);
      const targetTensor = createTarget(labels);
      
      trainData.push(imgTensor);
      trainLabels.push(targetTensor);
    }
    
    if (trainData.length === 0) {
      alert('No valid UTK Face images found. Make sure filenames follow format: age_gender_race_date.jpg');
      setTraining(false);
      return;
    }

    // Concatenate all data
    const xs = tf.concat(trainData);
    const ys = tf.concat(trainLabels);
    
    // Clean up individual tensors
    trainData.forEach(t => t.dispose());
    trainLabels.forEach(t => t.dispose());

    // Train
    await newModel.fit(xs, ys, {
      epochs: 20,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          setEpoch(epoch + 1);
          setLoss(logs.loss.toFixed(4));
        }
      }
    });

    xs.dispose();
    ys.dispose();
    
    setModel(newModel);
    setTraining(false);
    alert('Training complete!');
  };

  // Predict on new image
  const predictImage = async () => {
    if (!model) {
      alert('Please train the model first');
      return;
    }

    const file = fileInputRef.current.files[0];
    if (!file) {
      alert('Please select an image');
      return;
    }

    const imgTensor = await preprocessImage(file);
    const pred = model.predict(imgTensor);
    const values = await pred.data();
    
    // Parse predictions
    const age = Math.round(values[0] * 100);
    const gender = values[1] > values[2] ? 'Male' : 'Female';
    const raceIdx = values.slice(3, 8).indexOf(Math.max(...values.slice(3, 8)));
    const races = ['White', 'Black', 'Asian', 'Indian', 'Others'];
    const race = races[raceIdx];
    
    setPrediction({ age, gender, race });
    
    // Show image preview
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
    
    imgTensor.dispose();
    pred.dispose();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">Face Description CNN</h1>
        <p className="text-gray-600 mb-8">Predicts age, gender, and ethnicity from facial images</p>
        
        {/* Training Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">1. Train Model</h2>
          <p className="text-sm text-gray-600 mb-4">
            Select UTK Face dataset images (format: age_gender_race_date.jpg)
          </p>
          
          <input
            ref={trainInputRef}
            type="file"
            multiple
            accept="image/*"
            className="mb-4 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
          />
          
          <button
            onClick={trainModel}
            disabled={training}
            className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
          >
            {training ? `Training... Epoch ${epoch}` : 'Start Training'}
          </button>
          
          {loss && (
            <p className="mt-3 text-sm text-gray-600">
              Current Loss: <span className="font-semibold">{loss}</span>
            </p>
          )}
        </div>

        {/* Prediction Section */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">2. Predict Face</h2>
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="mb-4 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100"
          />
          
          <button
            onClick={predictImage}
            disabled={!model}
            className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
          >
            Predict
          </button>

          {/* Results */}
          {prediction && (
            <div className="mt-6 grid md:grid-cols-2 gap-6">
              {imagePreview && (
                <div>
                  <h3 className="font-semibold text-gray-700 mb-2">Input Image</h3>
                  <img src={imagePreview} alt="Preview" className="rounded-lg shadow-md w-full" />
                </div>
              )}
              
              <div>
                <h3 className="font-semibold text-gray-700 mb-3">Predictions</h3>
                <div className="space-y-3">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <span className="text-gray-600">Age:</span>
                    <span className="ml-2 text-xl font-bold text-indigo-600">{prediction.age}</span>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <span className="text-gray-600">Gender:</span>
                    <span className="ml-2 text-xl font-bold text-indigo-600">{prediction.gender}</span>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <span className="text-gray-600">Ethnicity:</span>
                    <span className="ml-2 text-xl font-bold text-indigo-600">{prediction.race}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Instructions */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900 mb-2">Quick Start Guide:</h3>
          <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
            <li>Download UTK Face dataset images</li>
            <li>Select multiple training images (the more, the better!)</li>
            <li>Click "Start Training" and wait for completion</li>
            <li>Upload a test face image and click "Predict"</li>
          </ol>
        </div>
      </div>
    </div>
  );
}