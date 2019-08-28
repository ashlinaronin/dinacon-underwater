const webcamElement = document.getElementById('webcam');
const examplesFish = document.getElementById('examples-fish');
const examplesNoFish = document.getElementById('examples-nofish');
const classifier = knnClassifier.create();
const classes = ['fish', 'nofish'];
let net;

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
      navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
      navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({video: true},
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
}

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  await setupWebcam();

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


  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      document.getElementById('console').innerText = `
        prediction: ${classes[result.classIndex]}\n
        probability: ${result.confidences[result.classIndex]}
      `;
    }

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }

  // // Make a prediction through the model on our image.
  // const imgEl = document.getElementById('img');
  // const result = await net.classify(imgEl);
  console.log(result);
}

app();