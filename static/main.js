const modelOutputText = document.getElementById('modelOutput');
const imageUploadButton = document.getElementById('imageUploadButton');
const imagePreviewDiv = document.getElementById('imagePreview');
const imageInpFile = document.getElementById('image-input');
const modelPredictionConfidenceness = document.getElementById('modelOutputConfidencess')
const imagePreviewImage = imagePreviewDiv.querySelector('.image-preview-image');
const imagePreviewDefaultText = imagePreviewDiv.querySelector('.image-preview-default-text');
const fileReader = new FileReader();
const endpoint = 'http://127.0.0.1:8080/predictions/resnet-18';
const bar = {
    "1": "SPIDER-MAN",
    "2": "VENOM",
    "3": "AUNT MAY",
    "4": "SPIDER-GWEN",
    "5": "YODA",
    "6": "LUKE SKYWALKER",
    "7": "R2-D2",
    "8": "MACE WINDU",
    "9": "GENERAL GRIEVOUS",
    "10": "KYLO REN",
    "11": "THE MANDALORIAN",
    "12": "CARA DUNE",
    "13": "KLATOOINIAN RAIDER 1",
    "14": "KLATOOINIAN RAIDER 2",
    "15": "MYSTERIO",
    "16": "FIREFIGHTER",
    "17": "NIGHT MONKEY",
    "18": "HARRY POTTER",
    "19": "RON WEASLEY",
    "20": "BLACK WIDOW",
    "21": "YELENA BELOVA",
    "22": "TASKMASTER",
    "23": "CAPTAIN AMERICA",
    "24": "OUTRIDER 1",
    "25": "OUTRIDER 2",
    "26": "OWEN GRADY",
    "27": "TRACKER TRAQUEUR RASTREADOR",
    "28": "IRON MAN MK 1",
    "29": "IRON MAN MK 5",
    "30": "IRON MAN MK 41",
    "31": "IRON MAN MK 50",
    "32": "JANNAH",
    "33": "HAN SOLO",
    "34": "DARTH VADER",
    "35": "ANAKIN SKYWALKER (SCARRED)",
    "36": "EMPEROR PALPATINE"
}

function hideDefaultTextAndShowImageElement() {
    imagePreviewDefaultText.style.display = "none";
    imagePreviewImage.style.display = "block";
}

const getImageFile = (imageInpFile) => {
    const {files: imageFile} = imageInpFile;
    return imageFile;
};

const onButtonClick = async () => {
    const files = getImageFile(imageInpFile);
    if (!files) return;
    const imageFile = files[0];
    const [className, percentConfidenceness] = await getPredictions(imageFile);
    setModelOutputAndPredictionConfidencenessText(className, percentConfidenceness);
};

const getPredictions = async (image) => {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: image,
        });
        const JSONResponse = await response.json();
        const [IndexWithHighestConfidence, confidenceness] = getIndexWithHighestConfidence(JSONResponse);
        const className = convertClassIndexToClassName(IndexWithHighestConfidence);
        const percentConfidenceness = convertToPercentAndRound(confidenceness)
        return [className, `${percentConfidenceness}%`];
    } catch (error) {
        return error;
    }
}

const getIndexWithHighestConfidence = (predictionObject) => {
    let classIndexWithHighestConfidence;
    let tempHighestConfidence = 0;
    for (const [classIndex, confidenceScore] of Object.entries(predictionObject)) {
        if (confidenceScore > tempHighestConfidence) {
            tempHighestConfidence = confidenceScore;
            classIndexWithHighestConfidence = classIndex;
        }
    }
    const highestConfidence = predictionObject[classIndexWithHighestConfidence];
    return [classIndexWithHighestConfidence, highestConfidence];
}

const convertClassIndexToClassName = (classIndex) => {
    return bar[`${classIndex}`];
}

const convertToPercentAndRound = (number) => {
    const percentRoundedNumber = number.toPrecision(2) * 100;
    return percentRoundedNumber;
}

const clearModelOutputAndPredictionConfidencenessText = () => {
    modelOutputText.textContent = '';
    modelPredictionConfidenceness.textContent = '';
};

const setModelOutputAndPredictionConfidencenessText = (className, confidencenessPercent) => {
    modelOutputText.textContent = className;
    modelPredictionConfidenceness.textContent = `Model confidenceness: ${confidencenessPercent}`;
}

const onImageInpChange = (files) => {
    const file = files[0];
    if (!file) return;
    hideDefaultTextAndShowImageElement();
    clearModelOutputAndPredictionConfidencenessText();
    fileReader.readAsDataURL(file);
};

const setImagePreviewsrc = (src) => {
    const { result } = src;
    imagePreviewImage.setAttribute('src', result);
};

fileReader.addEventListener('load', function(){
    setImagePreviewsrc(this);
});

imageInpFile.addEventListener("change", function() {
    onImageInpChange(this.files);
});

imageUploadButton.addEventListener("click", onButtonClick);