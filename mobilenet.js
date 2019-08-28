const classifier = knnClassifier.create();
const classes = ['fish', 'nofish'];
const fishExamples = ["fish-examples/2019-08-26_0611.jpg", "fish-examples/2019-08-26_0615.jpg", "fish-examples/2019-08-26_0622.jpg"];
const noFishExamples = ["nofish-examples/2019-08-20_1907.jpg", "nofish-examples/2019-08-22_2045.jpg"];
const imagesToTest = ["images-to-test/2019-08-27_0611.jpg", "images-to-test/2019-08-26_0618.jpg"];

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

async function loadTestImages() {
  imagesToTest.forEach(async imageSrc => {
    const img = await loadImage(imageSrc);
    document.getElementById("images-under-test").appendChild(img);
    await testImage(img);
  });
}

async function testImage(img) {
  if (classifier.getNumClasses() > 0) {
    // Get the activation from mobilenet from the test image.
    const activation = net.infer(img, 'conv_preds');

    // Get the most likely class and confidences from the classifier module.
    const result = await classifier.predictClass(activation);

    const resultText = document.createElement('div');
    resultText.innerText = `
      prediction: ${classes[result.classIndex]}\n
      confidence: ${result.confidences[result.classIndex]}
    `;
    img.parentNode.insertBefore(resultText, img.nextSibling);
  }
}


async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  // train the model
  await trainModel();

  // test images
  await loadTestImages();
}

app();