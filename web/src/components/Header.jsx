import { Leaf } from 'lucide-react'

const Header = () => {
  return (
    <div className="text-center d-flex justify-content-center flex-column align-items-center pt-5">
      <div className="mb-4 d-flex align-items-center justify-content-center" style={styles.iconContainer}>
        <Leaf size={50} />
      </div>
      <h1>Clasificador de Residuos</h1>
      <span>Identifica el tipo de basura con IA. Sube una imagen o describe el residuo.</span>
    </div>
  )
}

const styles = {
  iconContainer: {
    backgroundColor: "rgba(21, 141, 0, 0.341)",
    width: "100px",
    height: "100px",
    borderRadius: "50%",
  },
}

export default Header
