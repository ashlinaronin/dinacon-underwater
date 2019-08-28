const classifier = knnClassifier.create();
const classes = ['fish', 'nofish'];
const fishExamples = ["fish-examples/2019-08-26_0611.jpg", "fish-examples/2019-08-26_0615.jpg", "fish-examples/2019-08-26_0622.jpg"];
const noFishExamples = ["nofish-examples/2019-08-20_1907.jpg", "nofish-examples/2019-08-22_2045.jpg"];
const imagesToTest = ["images-to-test/2019-08-27_0611.jpg"];

let net;

async function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src
  });
}

async function addExampleFromImageSrc(imgSrc, classIndex) {
  const img = await loadImage(imgSrc);

  // get the intermediate activation of MobileNet 'conv_preds' and pass that to the KNN classifier
  const activation = net.infer(img, 'conv_preds');

  // pass the intermediate activation to the classifier
  classifier.addExample(activation, classIndex);
}

async function trainModel() {
  const fishPromises = Promise.all(fishExamples.map(async imgSrc => {
    await addExampleFromImageSrc(imgSrc, 0);
  }));

  const noFishPromises = Promise.all(noFishExamples.map(async imgSrc => {
    await addExampleFromImageSrc(imgSrc, 1);
  }));

  return Promise.all([fishPromises, noFishPromises]);
}


async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  await trainModel();

  const imgEl = document.getElementById('img-under-test');

  if (classifier.getNumClasses() > 0) {
    // Get the activation from mobilenet from the webcam.
    const activation = net.infer(imgEl, 'conv_preds');
    // Get the most likely class and confidences from the classifier module.
    const result = await classifier.predictClass(activation);

    document.getElementById('console').innerText = `
      prediction: ${classes[result.classIndex]}\n
      probability: ${result.confidences[result.classIndex]}
    `;
  }
}

app();