import { Card, Spinner } from "react-bootstrap";
import { BottleWine, Box, Handbag, NotebookText, Settings } from "lucide-react";

import capitalize from "../utils/capitalize";

import { CARTON, METAL, PAPEL, PLASTICO, VIDRIO } from "../constants/labels";

const Classification = ({ isLoading, predict, probabilities, confidence }) => {
  if (!predict && !isLoading) return null;

  return (
    <Card className="shadow-sm mt-3 mx-auto p-5" style={{ maxWidth: 650 }}>
      {isLoading && <Spinner variant="success" className="mx-auto" />}

      {!isLoading && (
        <div className="d-flex align-items-center">
          <div style={glassIcon.iconContainer}>
            {predict === VIDRIO && <BottleWine size={28} color="white" />}
            {predict === CARTON && <Box size={28} color="white" />}
            {predict === PAPEL && <NotebookText size={28} color="white" />}
            {predict === PLASTICO && <Handbag size={28} color="white" />}
            {predict === METAL && <Settings size={28} color="white" />}
          </div>
          <h3 style={{ color: "rgba(0, 0, 0, .7)" }}>{capitalize(predict)}</h3>
        </div>
      )}
      {!isLoading && probabilities && (
        <div className="mt-4">
          <h5>Probabilidades:</h5>
          <ul>
            {Object.entries(probabilities).map(([label, prob]) => (
              <li key={label}>
                {capitalize(label)}: {prob}
              </li>
            ))}
          </ul>
          {confidence && <h5>Confianza: {confidence.toFixed(2)}%</h5>}
        </div>
      )}
    </Card>
  );
};

const glassIcon = {
  iconContainer: {
    backgroundColor: "rgba(0, 118, 141, 0.34)",
    width: "50px",
    height: "50px",
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    marginRight: "20px",
  },
};

export default Classification;
