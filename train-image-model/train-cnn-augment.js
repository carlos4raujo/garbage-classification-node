// train-cnn-augment.js
// Script para entrenar un modelo de clasificación de imágenes de basura
// usando TensorFlow.js con aumento de datos (data augmentation)

// Importar librerías necesarias
import * as tf from "@tensorflow/tfjs-node"; // TensorFlow.js para Node.js
import fs from "fs"; // Sistema de archivos para leer imágenes
import path from "path"; // Manejo de rutas de archivos

// Constantes de configuración del modelo y dataset
const IMAGE_SIZE = 100; // Tamaño al que se redimensionarán todas las imágenes (100x100 píxeles)
const DATA_DIR = "./data"; // Directorio donde están las carpetas de cada clase de basura
const MODEL_DIR = "./models/image-classification"; // Directorio donde se guardará el modelo entrenado
const TEST_IMAGE = "./test_images/image-1.jpg"; // Imagen de prueba para validar el modelo

// === 1. Obtener clases ===
// Lee el directorio de datos y extrae las carpetas (cada carpeta es una clase de basura)
// Por ejemplo: cardboard, glass, metal, paper, plastic
const clases = fs
  .readdirSync(DATA_DIR) // Lee todos los elementos en el directorio de datos
  .filter((d) => fs.statSync(path.join(DATA_DIR, d)).isDirectory()); // Filtra solo los directorios
console.log("📦 Clases encontradas:", clases);

// === 2. Cargar imágenes ===
// Función asíncrona que carga todas las imágenes del dataset
// y las convierte en tensores (arrays multidimensionales) para TensorFlow
async function loadDataset() {
  const images = []; // Array para almacenar tensores de imágenes
  const labels = []; // Array para almacenar las etiquetas (índice de clase)

  // Iterar sobre cada clase (cardboard=0, glass=1, metal=2, etc.)
  for (let i = 0; i < clases.length; i++) {
    const classDir = path.join(DATA_DIR, clases[i]); // Ruta completa a la carpeta de la clase
    const files = fs.readdirSync(classDir); // Leer todos los archivos de la carpeta

    // Procesar cada archivo de imagen en la carpeta
    for (const file of files) {
      const filePath = path.join(classDir, file); // Ruta completa del archivo
      const buffer = fs.readFileSync(filePath); // Leer el archivo como buffer binario

      // Decodificar y preprocesar la imagen
      const imgTensor = tf.node
        .decodeImage(buffer, 1) // Decodificar imagen en escala de grises (1 canal)
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]) // Redimensionar a 100x100 píxeles
        .toFloat() // Convertir a números flotantes
        .div(255.0); // Normalizar píxeles de [0-255] a [0-1]

      images.push(imgTensor); // Agregar tensor de imagen al array
      labels.push(i); // Agregar índice de clase (0, 1, 2, etc.)
    }
  }

  // Convertir arrays de tensores a tensores apilados
  return {
    images: tf.stack(images), // Tensor 4D: [num_images, height, width, channels]
    labels: tf.tensor1d(labels, "int32"), // Tensor 1D: [num_images] con índices de clase
  };
}

// Cargar el dataset inicial
console.log("⏳ Cargando dataset...");
const { images, labels } = await loadDataset();
console.log(`✅ ${images.shape[0]} imágenes cargadas`);

// === 3. Función de aumento de datos ===
// El aumento de datos (data augmentation) crea versiones modificadas de las imágenes
// para aumentar artificialmente el tamaño del dataset y mejorar la generalización del modelo
function augmentImage(img) {
  // img es un tensor de forma [height, width, channels]
  let out = img.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]); // Añadir dimensión de batch

  // Flip horizontal aleatorio (voltear imagen horizontalmente con 50% de probabilidad)
  // Esto ayuda al modelo a reconocer objetos independientemente de su orientación
  if (Math.random() > 0.5) {
    out = tf.image.flipLeftRight(out);
  }

  // Random crop y resize (recorte y redimensionamiento aleatorio)
  // Simula zoom aleatorio entre 0.85x y 1.0x del tamaño original
  const zoomFactor = 0.85 + Math.random() * 0.15; // Factor de zoom entre 0.85 y 1.0
  const cropSize = Math.floor(IMAGE_SIZE * zoomFactor); // Tamaño del recorte
  const offsetY = Math.floor(Math.random() * (IMAGE_SIZE - cropSize)); // Offset vertical aleatorio
  const offsetX = Math.floor(Math.random() * (IMAGE_SIZE - cropSize)); // Offset horizontal aleatorio

  // Recortar y redimensionar la imagen
  out = tf.image.cropAndResize(
    out,
    [
      [
        offsetY / IMAGE_SIZE, // Coordenada Y superior normalizada
        offsetX / IMAGE_SIZE, // Coordenada X izquierda normalizada
        (offsetY + cropSize) / IMAGE_SIZE, // Coordenada Y inferior normalizada
        (offsetX + cropSize) / IMAGE_SIZE, // Coordenada X derecha normalizada
      ],
    ],
    [0], // Índice del batch
    [IMAGE_SIZE, IMAGE_SIZE] // Tamaño final después del resize
  );

  // Ajuste de brillo aleatorio multiplicando por un factor entre 0.85 y 1.15
  // Esto simula diferentes condiciones de iluminación
  const brightnessFactor = 0.85 + Math.random() * 0.3; // Factor de brillo
  out = out.mul(brightnessFactor).clipByValue(0, 1); // Multiplicar y mantener valores entre 0 y 1

  // Devolver el tensor con forma original [IMAGE_SIZE, IMAGE_SIZE, 1]
  return out.reshape([IMAGE_SIZE, IMAGE_SIZE, 1]);
}

// === 4. Generar dataset aumentado ===
// Esta función aplica aumento de datos a todas las imágenes del dataset original
// y combina las imágenes originales con las aumentadas para duplicar el tamaño del dataset
function augmentDataset(images, labels) {
  const augmentedImages = []; // Array para las imágenes aumentadas
  const augmentedLabels = []; // Array para las etiquetas de las imágenes aumentadas

  const num = images.shape[0]; // Número total de imágenes originales

  // Procesar cada imagen del dataset original
  for (let i = 0; i < num; i++) {
    // Extraer una sola imagen del tensor 4D
    const img = images
      .slice([i, 0, 0, 0], [1, IMAGE_SIZE, IMAGE_SIZE, 1]) // Extraer imagen i
      .squeeze(); // Eliminar dimensión de batch (de [1,100,100,1] a [100,100,1])
    const label = labels.arraySync()[i]; // Obtener la etiqueta de clase para esta imagen

    // Aplicar transformaciones de aumento de datos
    const augmented = augmentImage(img);
    augmentedImages.push(augmented); // Guardar imagen aumentada
    augmentedLabels.push(label); // Guardar la misma etiqueta
  }

  // Combinar imágenes originales y aumentadas en un solo dataset
  const allImages = tf.concat([images, tf.stack(augmentedImages)]); // Concatenar tensores
  const allLabels = tf.concat([labels, tf.tensor1d(augmentedLabels, "int32")]); // Concatenar etiquetas
  return { allImages, allLabels };
}

// Aplicar aumento de datos al dataset original
console.log("🔁 Aplicando aumento de datos...");
const { allImages, allLabels } = augmentDataset(images, labels);
console.log(`✨ Dataset aumentado: ${allImages.shape[0]} imágenes totales`);

// === 5. Convertir labels a one-hot encoding ===
// One-hot encoding convierte índices de clase (0,1,2,3,4) a vectores binarios
// Por ejemplo: clase 2 con 5 clases → [0, 0, 1, 0, 0]
// Esto es necesario para la función de pérdida categorical crossentropy
const allLabelsOneHot = tf.oneHot(allLabels, clases.length);

// === 6. Dividir dataset (85% - 15%) ===
// Separar el dataset en conjunto de entrenamiento y validación
// Training: 85% - para entrenar el modelo
// Validation: 15% - para evaluar el rendimiento durante el entrenamiento
const total = allImages.shape[0]; // Total de imágenes en el dataset aumentado
const trainSize = Math.floor(total * 0.85); // 85% para entrenamiento
const valSize = total - trainSize; // 15% para validación

// Dividir imágenes en train y validation
const [imagesTrain, imagesVal] = tf.split(allImages, [trainSize, valSize]);
// Dividir etiquetas en train y validation
const [labelsTrain, labelsVal] = tf.split(allLabelsOneHot, [
  trainSize,
  valSize,
]);
console.log(`📊 Train: ${trainSize}, Validación: ${valSize}`);

// === 7. Crear modelo CNN (Red Neuronal Convolucional) ===
// El modelo Sequential permite apilar capas una tras otra
const model = tf.sequential();

// Primera capa convolucional
// - inputShape: [100, 100, 1] - imágenes de 100x100 píxeles en escala de grises
// - filters: 32 - aprende 32 filtros/características diferentes
// - kernelSize: 3 - cada filtro es de 3x3 píxeles
// - activation: 'relu' - función de activación ReLU (introduce no-linealidad)
model.add(
  tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
    filters: 32,
    kernelSize: 3,
    activation: "relu",
  })
);
// Capa de max pooling - reduce dimensiones tomando el valor máximo en ventanas de 2x2
// Esto reduce el tamaño de la imagen a la mitad y ayuda a detectar características invariantes a la posición
model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

// Segunda capa convolucional
// - filters: 64 - aprende características más complejas con 64 filtros
model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
// Segunda capa de max pooling
model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

// Tercera capa convolucional
// - filters: 128 - aprende características aún más abstractas con 128 filtros
model.add(
  tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" })
);
// Tercera capa de max pooling
model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

// Capa flatten - convierte el tensor 3D en un vector 1D para las capas densas
model.add(tf.layers.flatten());

// Capa de dropout - desactiva aleatoriamente 30% de las neuronas durante entrenamiento
// Esto previene el overfitting (sobreajuste) al forzar al modelo a no depender de neuronas específicas
model.add(tf.layers.dropout({ rate: 0.3 }));

// Capa densa (fully connected) con 100 neuronas
// Aprende combinaciones complejas de las características extraídas por las capas convolucionales
model.add(tf.layers.dense({ units: 100, activation: "relu" }));

// Capa de salida - una neurona por cada clase
// - units: número de clases (cardboard, glass, metal, paper, plastic)
// - activation: 'softmax' - convierte las salidas en probabilidades que suman 1
model.add(tf.layers.dense({ units: clases.length, activation: "softmax" }));

// Compilar el modelo con configuración de optimización y métricas
model.compile({
  optimizer: tf.train.adam(), // Optimizador Adam - algoritmo de descenso de gradiente adaptativo
  loss: "categoricalCrossentropy", // Función de pérdida para clasificación multiclase
  metrics: ["accuracy"], // Métrica a monitorear: precisión (accuracy)
});

// === 8. Entrenamiento ===
// Entrenar el modelo con los datos de entrenamiento
console.log("🚀 Entrenando modelo...");
await model.fit(imagesTrain, labelsTrain, {
  epochs: 30, // Número de épocas - cuántas veces el modelo verá todo el dataset (ajustable)
  batchSize: 32, // Tamaño del batch - procesa 32 imágenes a la vez antes de actualizar pesos
  validationData: [imagesVal, labelsVal], // Datos de validación para evaluar en cada época
  shuffle: true, // Mezclar datos en cada época para evitar sesgos de orden
  callbacks: {
    // Callback que se ejecuta al final de cada época
    onEpochEnd: (epoch, logs) => {
      // Mostrar progreso: número de época, pérdida y precisión en validación
      console.log(
        `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, val_acc=${(
          logs.val_acc ||
          logs.val_accuracy ||
          0
        ).toFixed(4)}`
      );
    },
  },
});

// === 9. Evaluación ===
// Evaluar el modelo entrenado con el conjunto de validación
const evalResult = model.evaluate(imagesVal, labelsVal);
// Obtener los valores de pérdida y precisión
const [lossArray, accArray] = await Promise.all(
  evalResult.map((x) => x.data())
);
const loss = lossArray[0]; // Pérdida final
const acc = accArray[0]; // Precisión final
console.log(`📉 Pérdida: ${loss?.toFixed(4)}, Precisión: ${acc?.toFixed(4)}`);

// === 10. Guardar modelo ===
// Guardar el modelo entrenado en el sistema de archivos
// El modelo se guarda en formato JSON junto con los pesos en archivos binarios
await model.save(`file://${MODEL_DIR}`);
console.log(`✅ Modelo guardado en ${MODEL_DIR}`);

// === 11. Predicción ===
// Función para hacer predicciones sobre nuevas imágenes
async function predictImage(imgPath) {
  const buffer = fs.readFileSync(imgPath); // Leer imagen de prueba

  // Preprocesar la imagen de la misma manera que las imágenes de entrenamiento
  const img = tf.node
    .decodeImage(buffer, 1) // Decodificar en escala de grises
    .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]) // Redimensionar a 100x100
    .toFloat() // Convertir a flotante
    .div(255.0) // Normalizar a [0, 1]
    .expandDims(0); // Añadir dimensión de batch: [1, 100, 100, 1]

  // Hacer la predicción
  const pred = model.predict(img); // Obtener probabilidades para cada clase
  const labelIndex = pred.argMax(-1).dataSync()[0]; // Obtener índice de la clase con mayor probabilidad
  console.log(`🖼️ Clase predicha: ${clases[labelIndex]}`); // Mostrar nombre de la clase
  return clases[labelIndex]; // Devolver el nombre de la clase
}

// Probar el modelo con una imagen de prueba
await predictImage(TEST_IMAGE);
