import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

const model = await tf.loadLayersModel('file://./model/model.json');
const wordIndex = JSON.parse(fs.readFileSync('./word_index.json', 'utf8'));
const labels = JSON.parse(fs.readFileSync('./labels.json', 'utf8'));

const maxLen = model.inputs[0].shape[1];

function preprocess(text) {
  const words = text.toLowerCase().split(/\s+/);
  const seq = words.map(w => wordIndex[w] || 0);
  const padded = new Array(maxLen).fill(0);
  for (let i = 0; i < Math.min(seq.length, maxLen); i++) padded[i] = seq[i];
  return tf.tensor2d([padded]);
}

async function classify(text) {
  const input = preprocess(text);
  const pred = model.predict(input);
  const data = await pred.data();
  const idx = data.indexOf(Math.max(...data));
  console.log(`Predicción: ${labels[idx]}`);
}

await classify('botella rota');
