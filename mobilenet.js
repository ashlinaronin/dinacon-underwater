const examplesFish = document.getElementById('examples-fish');
const examplesNoFish = document.getElementById('examples-nofish');
const classifier = knnClassifier.create();
const classes = ['fish', 'nofish'];
let net;

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  Array.from(examplesFish.children).forEach(el => {
    // get the intermediate activation of MobileNet 'conv_preds' and pass that to the KNN classifier
    const activation = net.infer(el, 'conv_preds');

    // pass the intermediate activation to the classifier
    classifier.addExample(activation, 0);
  });

  Array.from(examplesNoFish.children).forEach(el => {
    // get the intermediate activation of MobileNet 'conv_preds' and pass that to the KNN classifier
    const activation = net.infer(el, 'conv_preds');

    // pass the intermediate activation to the classifier
    classifier.addExample(activation, 1);
  });

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