import * as tf from "@tensorflow/tfjs";
import { Button, Form, FormLabel } from "react-bootstrap";

import { CARTON, METAL, PAPEL, PLASTICO, VIDRIO } from "../constants/labels";

import vocab from "../assets/word_index.json";

const LABELS = [CARTON, VIDRIO, METAL, PAPEL, PLASTICO];
// const LABELS = [VIDRIO, PLASTICO, METAL, PAPEL, CARTON];

const TextClassifier = ({
  show,
  model,
  setPredict,
  setIsLoadingPredict,
  setProbabilities,
  setConfidence,
}) => {
  if (!show) return null;

  const preprocess = (text) => {
    setIsLoadingPredict(true);
    setPredict(null);
    const maxLen = model.inputs[0].shape[1];
    const words = text.toLowerCase().split(/\s+/);
    const seq = words.map((w) => vocab[w] || 0);
    const padded = new Array(maxLen).fill(0);
    for (let i = 0; i < Math.min(seq.length, maxLen); i++) padded[i] = seq[i];
    return tf.tensor2d([padded]);
  };

  const predictText = async (text) => {
    const input = preprocess(text);
    const data = await model.predict(input).data();
    const index = data.indexOf(Math.max(...data));
    const confidence = Math.max(...data) * 100;

    setProbabilities({
      carton: `${(data[0] * 100).toFixed(2)}%`,
      vidrio: `${(data[1] * 100).toFixed(2)}%`,
      metal: `${(data[2] * 100).toFixed(2)}%`,
      papel: `${(data[3] * 100).toFixed(2)}%`,
      plastico: `${(data[4] * 100).toFixed(2)}%`,
    });
    setConfidence(confidence);

    setIsLoadingPredict(false);
    setPredict(LABELS[index]);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const textInput = formData.get("textInput");
    predictText(textInput);
  };

  return (
    <Form onSubmit={handleSubmit}>
      <FormLabel>Describe el residuo</FormLabel>
      <Form.Control
        type="text"
        placeholder="Ingresa un objeto"
        name="textInput"
      />
      <Button
        size="sm"
        variant="success"
        style={{ width: "100%", rowGap: 12 }}
        type="submit"
        className="d-flex justify-content-center my-4 px-4"
      >
        <span>Clasificar residuo</span>
      </Button>
    </Form>
  );
};

export default TextClassifier;
