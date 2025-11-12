// Importación de dependencias necesarias
import * as tf from "@tensorflow/tfjs-node"; // TensorFlow.js para Node.js - permite entrenar modelos en el servidor
import fs from "fs"; // File System - para leer archivos y directorios
import path from "path"; // Utilidades para manejar rutas de archivos

// ========== CONFIGURACIÓN DEL MODELO ==========
// Tamaño al que se redimensionarán todas las imágenes (100x100 píxeles)
// Es importante que todas las imágenes tengan el mismo tamaño para entrenar la red neuronal
const IMAGE_SIZE = 100;

// Directorio donde se encuentran las carpetas con las imágenes de entrenamiento
// Cada subcarpeta representa una clase (cardboard, glass, metal, paper, plastic)
const DATA_DIR = "./data";

// ========== IDENTIFICACIÓN DE CLASES ==========
// Lee todas las subcarpetas dentro de DATA_DIR
// Cada carpeta representa una categoría de basura (cardboard, glass, metal, paper, plastic)
// filter() asegura que solo se tomen carpetas, no archivos sueltos
const clases = fs
  .readdirSync(DATA_DIR)
  .filter((d) => fs.statSync(path.join(DATA_DIR, d)).isDirectory());

// ========== FUNCIÓN PARA CARGAR Y PREPROCESAR IMÁGENES ==========
/**
 * Carga todas las imágenes del dataset y las convierte en tensores
 * También genera las etiquetas correspondientes para cada imagen
 * @returns {Object} Objeto con dos tensores: images (datos) y labels (etiquetas)
 */
async function loadImages() {
  const images = []; // Array temporal para almacenar tensores de imágenes
  const labels = []; // Array temporal para almacenar las etiquetas numéricas

  // Iterar sobre cada clase (cardboard=0, glass=1, metal=2, etc.)
  for (let i = 0; i < clases.length; i++) {
    const classDir = path.join(DATA_DIR, clases[i]); // Ruta completa a la carpeta de la clase
    const files = fs.readdirSync(classDir); // Lista de archivos de imagen en la carpeta

    // Procesar cada archivo de imagen en la carpeta actual
    for (const file of files) {
      const imgPath = path.join(classDir, file); // Ruta completa del archivo
      const imgBuffer = fs.readFileSync(imgPath); // Leer imagen como buffer binario

      // Convertir imagen a tensor y preprocesarla:
      const imgTensor = tf.node
        .decodeImage(imgBuffer, 1) // Decodificar imagen (1 = escala de grises para reducir complejidad)
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]) // Redimensionar a 100x100 píxeles
        .toFloat() // Convertir valores a números decimales
        .div(255.0); // Normalizar valores de 0-255 a 0-1 (ayuda al entrenamiento)

      images.push(imgTensor); // Agregar tensor procesado al array
      labels.push(i); // Agregar índice de clase como etiqueta (0, 1, 2, 3, 4)
    }
  }

  // Retornar tensores apilados listos para entrenar
  return {
    images: tf.stack(images), // Apilar todos los tensores en un solo tensor 4D
    labels: tf.tensor1d(labels, "float32"), // Convertir array de etiquetas a tensor 1D
  };
}

// Ejecutar la función de carga de imágenes
const { images, labels } = await loadImages();

// ========== CONSTRUCCIÓN DE LA RED NEURONAL CONVOLUCIONAL (CNN) ==========
// Crear un modelo secuencial (capas apiladas una tras otra)
const model = tf.sequential();

// BLOQUE CONVOLUCIONAL 1: Detección de características básicas
model.add(
  tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1], // Forma de entrada: 100x100 píxeles, 1 canal (escala de grises)
    filters: 32, // Número de filtros/kernels que detectan patrones (bordes, líneas, etc.)
    kernelSize: 3, // Tamaño del filtro: 3x3 píxeles
    activation: "relu", // Función de activación ReLU (introduce no-linealidad)
  })
);
// Pooling: Reduce dimensionalidad a la mitad, conserva características importantes
model.add(tf.layers.maxPooling2d({ poolSize: 2 })); // Toma el valor máximo de cada región 2x2

// BLOQUE CONVOLUCIONAL 2: Detección de características intermedias
model.add(
  tf.layers.conv2d({
    filters: 64, // Más filtros para detectar patrones más complejos
    kernelSize: 3,
    activation: "relu",
  })
);
model.add(tf.layers.maxPooling2d({ poolSize: 2 })); // Segunda reducción de dimensionalidad

// BLOQUE CONVOLUCIONAL 3: Detección de características avanzadas
model.add(
  tf.layers.conv2d({
    filters: 128, // Aún más filtros para características de alto nivel
    kernelSize: 3,
    activation: "relu",
  })
);
model.add(tf.layers.maxPooling2d({ poolSize: 2 })); // Tercera reducción de dimensionalidad

// CAPAS DENSAS (Fully Connected): Clasificación final
model.add(tf.layers.flatten()); // Convertir datos 2D en vector 1D para capas densas

// Dropout: Previene overfitting desactivando aleatoriamente 30% de neuronas durante entrenamiento
model.add(tf.layers.dropout({ rate: 0.3 }));

// Capa densa intermedia: Aprende combinaciones de características
model.add(tf.layers.dense({ units: 100, activation: "relu" }));

// Capa de salida: Una neurona por cada clase, softmax para obtener probabilidades
model.add(
  tf.layers.dense({
    units: clases.length, // Número de clases (5: cardboard, glass, metal, paper, plastic)
    activation: "softmax", // Convierte outputs en probabilidades que suman 1
  })
);

// ========== COMPILACIÓN DEL MODELO ==========
// Configurar cómo se entrenará el modelo
model.compile({
  optimizer: tf.train.adam(), // Algoritmo Adam: ajusta pesos de manera eficiente
  loss: "sparseCategoricalCrossentropy", // Función de pérdida para clasificación multiclase con etiquetas numéricas
  metrics: ["accuracy"], // Métrica a monitorear: precisión del modelo
});

// ========== ENTRENAMIENTO DEL MODELO ==========
// Entrenar el modelo con las imágenes cargadas
await model.fit(images, labels, {
  epochs: 30, // Número de veces que el modelo verá todo el dataset
  batchSize: 32, // Número de imágenes procesadas antes de actualizar pesos
  validationSplit: 0.15, // 15% de datos reservados para validación (no se usan en entrenamiento)
  shuffle: true, // Mezclar datos en cada época para mejor generalización
});

// ========== GUARDAR MODELO ENTRENADO ==========
// Guardar el modelo en formato compatible con navegadores web (TensorFlow.js)
// Se guardarán archivos model.json (arquitectura) y weight.bin (pesos entrenados)
await model.save("file://./models/image-classification");
console.log("✅ Modelo guardado en ./models/image-classification");
