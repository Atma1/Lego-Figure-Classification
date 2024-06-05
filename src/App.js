import './styles.css';
import Body from './components/Body';
import Navbar from './components/Navbar';
import React from 'react';

const endpoint = "http://localhost:5000/predict"

export default function App() {
  const [file, setFile] = React.useState();
  const [prediction, setPrediction] = React.useState([]);

  const handleFileInputChange = (e) => {
    setFile(e.target.files[0]);
  }

  const getPrediction = async (image) => {
    let formData = new FormData();
    formData.append('image', image)

    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
  });
    console.log(response);
    return await response.json();
  }

  const handleUploadClick = async (image) => {
    if (file) {
      try {
        const res = await getPrediction(image);
        console.log(res);
        setPrediction(res);
      } catch (error) {
        console.error(error);
      } 
    }
  }

  return (
    <>
      <Navbar />
      <Body 
      file={file}
      handleFileInputChange={handleFileInputChange}
      handleUploadClick={handleUploadClick}
      prediction={prediction}
      />
    </>
  );
}
