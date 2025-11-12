import { useCallback, useEffect, useState } from "react";

import * as tf from "@tensorflow/tfjs";
import { CloudUpload } from "lucide-react";
import { useDropzone } from "react-dropzone";
import { Button } from "react-bootstrap";
import { CARTON, METAL, PAPEL, PLASTICO, VIDRIO } from "../constants/labels";

const labels = [CARTON, VIDRIO, METAL, PAPEL, PLASTICO];

const ImageClassifier = ({
  show,
  model,
  setIsLoadingPredict,
  setPredict,
  setProbabilities,
  setConfidence,
}) => {
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();

  const onDrop = useCallback((acceptedFiles) => {
    console.log(acceptedFiles);
    if (!acceptedFiles || acceptedFiles.length === 0) {
      setSelectedFile(undefined);
      return;
    }

    setSelectedFile(acceptedFiles[0]);
  }, []);

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  const preprocessImage = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");
          const IMAGE_SIZE = 100;
          canvas.width = IMAGE_SIZE;
          canvas.height = IMAGE_SIZE;
          ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
          const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
          const imgTensor = tf.browser
            .fromPixels(imageData, 1) // 1 = grayscale
            .toFloat()
            .div(255.0)
            .expandDims(0); // Añadir dimensión de batch
          resolve(imgTensor);
        };
        img.onerror = (error) => reject(error);
        img.src = reader.result;
      };
      reader.onerror = (error) => reject(error);
      reader.readAsDataURL(file);
    });
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }

    setIsLoadingPredict(true);

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);

    let prediction;

    preprocessImage(selectedFile)
      .then(async (tensor) => {
        prediction = model.predict(tensor);
        return prediction.data();
      })
      .then((probabilities) => {
        const predictedClass = prediction.argMax(-1).dataSync()[0];
        const confidence = probabilities[predictedClass];

        setIsLoadingPredict(false);

        setConfidence(confidence * 100);
        setProbabilities({
          carton: `${(probabilities[0] * 100).toFixed(2)}%`,
          vidrio: `${(probabilities[1] * 100).toFixed(2)}%`,
          metal: `${(probabilities[2] * 100).toFixed(2)}%`,
          papel: `${(probabilities[3] * 100).toFixed(2)}%`,
          plastico: `${(probabilities[4] * 100).toFixed(2)}%`,
        });
        setPredict(labels[predictedClass]);
      });

    return () => URL.revokeObjectURL(objectUrl);
  }, [
    selectedFile,
    setIsLoadingPredict,
    model,
    setPredict,
    setConfidence,
    setProbabilities,
  ]);

  const onClearImage = () => {
    setSelectedFile(undefined);
    setPreview(undefined);
    setPredict(null);
    setProbabilities(null);
    setConfidence(null);
  };

  if (!show) return null;

  return (
    <>
      {preview ? (
        <>
          <div
            style={{
              width: 400,
              height: 400,
              margin: "0px auto 16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              overflow: "hidden",
              borderRadius: 8,
              backgroundImage: `url(${preview})`,
              backgroundSize: "contain",
              backgroundRepeat: "no-repeat",
              backgroundPosition: "center",
            }}
          />
          <Button
            size="sm"
            color="info"
            className="my-4 mx-auto d-block"
            onClick={onClearImage}
          >
            Limpiar
          </Button>
        </>
      ) : (
        <div {...getRootProps()} className="cursor-pointer">
          <input {...getInputProps()} />

          <div
            className="dropzone d-flex flex-column align-items-center justify-content-center"
            style={{ cursor: "pointer" }}
          >
            <CloudUpload size={36} />
            <span className="fs-5 fw-semibold">
              Haz click para subir o arrastra una imagen
            </span>
            <span className="fs-6 opacity-75">PNG, JPG, WEBP hasta 10MB</span>
          </div>
        </div>
      )}
    </>
  );
};

export default ImageClassifier;
