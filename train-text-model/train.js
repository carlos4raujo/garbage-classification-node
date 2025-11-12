// Importamos TensorFlow.js versión Node para procesamiento de machine learning
import * as tf from '@tensorflow/tfjs-node';
// Importamos el módulo de sistema de archivos para leer/escribir archivos
import fs from 'fs';

// === 1. Cargar y Preparar el Dataset ===
// Leemos el archivo JSON que contiene nuestros datos de entrenamiento
const rawData = JSON.parse(fs.readFileSync('./dataset.json', 'utf8'));

// Inicializamos arrays para almacenar los textos y sus etiquetas
const texts = [];      // Almacenará los textos de ejemplo
const labels = [];     // Almacenará las etiquetas correspondientes
// Extraemos los nombres de las categorías del dataset
const labelNames = Object.keys(rawData);

// Procesamos cada ejemplo del dataset
for (const [label, examples] of Object.entries(rawData)) {
  for (const ex of examples) {
    // Reemplazamos los guiones bajos por espacios y añadimos el texto
    texts.push(ex.replace(/_/g, ' '));
    // Guardamos la etiqueta correspondiente
    labels.push(label);
  }
}

// === 2. Crear y Configurar el Tokenizador ===
// Creamos un conjunto (Set) para almacenar el vocabulario único
const vocab = new Set();
// Procesamos cada texto, lo dividimos en palabras y añadimos cada palabra al vocabulario
texts.forEach(t => t.split(/\s+/).forEach(w => vocab.add(w)));
// Creamos un índice numérico para cada palabra (empezando desde 1, 0 se reserva para padding)
const wordIndex = {};
Array.from(vocab).forEach((w, i) => (wordIndex[w] = i + 1));

// Convertimos los textos en secuencias numéricas usando el índice de palabras
const sequences = texts.map(t =>
  t.split(/\s+/).map(w => wordIndex[w] || 0)
);

// Encontramos la longitud máxima de las secuencias para hacer padding
const maxLen = Math.max(...sequences.map(seq => seq.length));
// Aplicamos padding a todas las secuencias para que tengan la misma longitud
const padded = sequences.map(seq => {
  const arr = new Array(maxLen).fill(0); // Rellenamos con ceros
  seq.forEach((n, i) => (arr[i] = n));   // Copiamos la secuencia original
  return arr;
});

// === 3. Crear Tensores para el Entrenamiento ===
// Convertimos las secuencias con padding a un tensor 2D
const X = tf.tensor2d(padded);
// Convertimos las etiquetas a índices numéricos
const yIdx = labels.map(l => labelNames.indexOf(l));
// Convertimos las etiquetas a formato one-hot encoding
const y = tf.oneHot(tf.tensor1d(yIdx, 'int32'), labelNames.length);

// === 4. Definir la Arquitectura del Modelo ===
// Creamos un modelo secuencial (capas en serie)
const model = tf.sequential();
// Añadimos una capa de embedding para convertir palabras en vectores densos
model.add(tf.layers.embedding({
  inputDim: vocab.size + 1,    // Tamaño del vocabulario + 1 para el token de padding
  outputDim: 16,               // Dimensión del espacio de embedding
  inputLength: maxLen          // Longitud de las secuencias de entrada
}));
// Añadimos una capa de pooling global para reducir dimensionalidad
model.add(tf.layers.globalAveragePooling1d());
// Añadimos una capa densa con activación ReLU
model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
// Capa de salida con softmax para clasificación multiclase
model.add(tf.layers.dense({ units: labelNames.length, activation: 'softmax' }));

// Compilamos el modelo con la configuración de entrenamiento
model.compile({
  loss: 'categoricalCrossentropy',  // Función de pérdida para clasificación multiclase
  optimizer: 'adam',                // Optimizador Adam
  metrics: ['accuracy']             // Métrica para monitorear el entrenamiento
});

// === 5. Entrenar el Modelo ===
// Iniciamos el entrenamiento con los parámetros especificados
await model.fit(X, y, {
  epochs: 200,    // Número de iteraciones sobre todo el dataset
  verbose: 1      // Nivel de detalle en los logs de entrenamiento
});

// === 6. Guardar el Modelo y Datos Auxiliares ===
// Guardamos el modelo entrenado
await model.save('file://./model');

// Guardamos el índice de palabras para usar en predicciones
fs.writeFileSync('./word_index.json', JSON.stringify(wordIndex, null, 2));
// Guardamos los nombres de las etiquetas
fs.writeFileSync('./labels.json', JSON.stringify(labelNames, null, 2));

// Indicamos que el proceso ha finalizado exitosamente
console.log('✅ Modelo entrenado y guardado en ./model');
