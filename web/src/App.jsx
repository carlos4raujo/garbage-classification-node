import { useEffect, useRef, useState } from "react";

import * as tf from "@tensorflow/tfjs";
import { Card, Container, Button, Row, Col } from "react-bootstrap";

import Header from "./components/Header";
import ImageClassifier from "./components/ImageClassifier";
import TextClassifier from "./components/TextClassifier";

import Classification from "./components/Classification";

function App() {
  const [type, setType] = useState("image");
  const [textClassificationModel, setTextClassificationModel] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [predict, setPredict] = useState(null);
  const [isLoadingPredict, setIsLoadingPredict] = useState(null);
  const [imageClassificationModel, setImageClassificationModel] =
    useState(null);
  const isTextModelLoaded = useRef(false);
  const isImageModelLoaded = useRef(false);

  useEffect(() => {
    const loadModel = async () => {
      if (isTextModelLoaded.current) return;
      try {
        const model = await tf.loadLayersModel(
          "/assets/models/text-classification/model.json"
        );
        setTextClassificationModel(model);
        console.log("✅ Modelo cargado");
        isTextModelLoaded.current = true;
      } catch (error) {
        console.error("❌ Error al cargar el modelo:", error);
      }
    };
    loadModel();
  }, []);

  useEffect(() => {
    const loadModel = async () => {
      if (isImageModelLoaded.current) return;
      try {
        const model = await tf.loadLayersModel(
          // "/assets/models/image-classification/model.json"
          "/assets/models/cnn-augment-model/model.json"
        );
        setImageClassificationModel(model);
        console.log("✅ Modelo de imagen cargado");
      } catch (error) {
        console.error("❌ Error al cargar el modelo de imagen:", error);
      }
    };
    loadModel();
  }, []);

  return (
    <div
      style={{
        backgroundColor: "oklch(.98 .005 120)",
        height: "100%",
        paddingBottom: 50,
      }}
    >
      <Header />
      <Container>
        <Card className="shadow-sm mt-3 mx-auto" style={{ maxWidth: 650 }}>
          <div className="p-3">
            <Row className="mb-4">
              <Col md="6">
                <Button
                  variant="success"
                  size="sm"
                  style={{ width: "100%" }}
                  disabled={!imageClassificationModel || type === "image"}
                  onClick={() => {
                    setType("image");
                    setPredict(null);
                    setProbabilities(null);
                    setConfidence(null);
                  }}
                >
                  Imagen
                </Button>
              </Col>
              <Col md="6">
                <Button
                  style={{ width: "100%" }}
                  size="sm"
                  variant="success"
                  disabled={!textClassificationModel || type === "text"}
                  onClick={() => {
                    setType("text");
                    setPredict(null);
                    setProbabilities(null);
                    setConfidence(null);
                  }}
                >
                  Texto
                </Button>
              </Col>
            </Row>
            <ImageClassifier
              model={imageClassificationModel}
              show={type === "image"}
              setIsLoadingPredict={setIsLoadingPredict}
              setPredict={setPredict}
              setProbabilities={setProbabilities}
              setConfidence={setConfidence}
            />
            <TextClassifier
              model={textClassificationModel}
              show={type === "text"}
              setIsLoadingPredict={setIsLoadingPredict}
              setPredict={setPredict}
              setProbabilities={setProbabilities}
              setConfidence={setConfidence}
            />
          </div>
        </Card>
        <Classification
          predict={predict}
          isLoading={isLoadingPredict}
          probabilities={probabilities}
          confidence={confidence}
        />
      </Container>
    </div>
  );
}

export default App;
