export default function Body({file, handleFileInputChange, handleUploadClick, prediction}) {
    return (
    <>
        <div className={"model-output-title"}>
            <h3 class="model-output" id="modelOutput">Prediction: {prediction.class}</h3>
            <h3 class="model-output-confidence" id="modelOutputConfidencess">Confidence: {prediction.confidence}</h3>
        </div>
        <div class="image-preview" id="imagePreview">
            {file && <img src={URL.createObjectURL(file)} alt='' class="image-preview-image"/>}
            {!file && <span class="image-preview-default-text" id="image-preview-text">No Image Selected</span>}
        </div>
        <div id="button-and-input-container">
            <label for="imagePreview" id="input-label">Choose Image To Classifiy</label>
            <input type="file" name="imagePreview" id="image-input" accept=".jpg, .jpeg" onChange={handleFileInputChange}/>
            <button type="submit" class="image-upload-button bg-white hover:bg-gray-200" id="imageUploadButton" onClick={() => handleUploadClick(file)}>Get Prediction</button>
        </div>
    </>
    );
}