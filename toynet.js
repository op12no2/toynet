//
// @colintomjenkins
//
// A Javascript version of @schienal's blog post titled
// "A worked example of backpropagation" at
// https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html
//
// If you use a folding editor the fold marks are {{{ and }}}.
//
// Node names from the blog post translated to this code:-
//
// x1 == neti[0]
// x1 == neti[1]
// h1 == neth[0]
// h2 == neth[1]
// y1 == neto[0]
// y2 == neto[1]
//
// You can create and train different (input, hidden output) fully
// connected nets to the one in the post using these globals:-
//

var netInputSize   = 2;  // input layer.
var netHiddenSize  = 2;  // hidden later.
var netOutputSize  = 2;  // output layer.

//{{{  build net

function netNode (weightsSize) {
  this.in          = 0;
  this.gin         = 0;
  this.out         = 0;
  this.gout        = 0;
  this.weights     = Array(weightsSize);
  this.gweights    = Array(weightsSize);
  this.gweightssum = Array(weightsSize);
}

var neti = Array(netInputSize);

var neth = Array(netHiddenSize);
for (var h=0; h < netHiddenSize; h++) {
  neth[h] = new netNode(netInputSize);
}

var neto = Array(netOutputSize);
for (var o=0; o < netOutputSize; o++) {
  neto[o] = new netNode(netHiddenSize);
}

//}}}

//{{{  sigmoid

function sigmoid(x) {
  return (1.0 / (1.0 + Math.exp(-x)));
}

function dsigmoid(x) {
  return sigmoid(x) * (1.0 - sigmoid(x));
}

//}}}
//{{{  netLoss

function netLoss(target) {

  var x = 0.0;

  for (var o=0; o < netOutputSize; o++) {
    x += (target[o] - neto[o].out) * (target[o] - neto[o].out);
  }

  return x;
}

//}}}
//{{{  netForward()

function netForward(inputs) {

  if (inputs.length != netInputSize) {
    console.log('netForward','input vector length must be',netInputSize,'your length is',input.length);
    process.exit;
  }

  for (var i=0; i < netInputSize; i++)
    neti[i] = inputs[i];

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    hidden.in = 0;
    for (var i=0; i < netInputSize; i++) {
      hidden.in += hidden.weights[i] * neti[i];
    }
    hidden.out = sigmoid(hidden.in);
  }

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    output.in = 0;
    for (var h=0; h < netHiddenSize; h++) {
      output.in += output.weights[h] * neth[h].out;
    }
    output.out = sigmoid(output.in);
  }
}

//}}}
//{{{  netInitWeights()

function netInitWeights() {

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    for (var i=0; i < netInputSize; i++) {
      hidden.weights[i] = Math.random();
    }
  }

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    for (var h=0; h < netHiddenSize; h++) {
      output.weights[h] = Math.random();
    }
  }
}

//}}}
//{{{  netCalcGradients()

function netCalcGradients(targets) {

  if (targets.length != netOutputSize) {
    console.log('netCallGradients','output vector length must be',netOutputSize,'your length is',targets.length);
    process.exit;
  }

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    output.gout = 2 * (output.out - targets[o]);
    output.gin  = dsigmoid(output.in) * output.gout;
  }

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    for (var h=0; h < netHiddenSize; h++) {
      var hidden = neth[h];
      output.gweights[h] = output.gin * hidden.out;
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    hidden.gout = 0;
    for (var o=0; o < netOutputSize; o++) {
      var output = neto[o];
      hidden.gout += output.gin * output.weights[h];
    }
    hidden.gin = dsigmoid(hidden.in) * hidden.gout;
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    for (var i=0; i < netInputSize; i++) {
      hidden.gweights[i] = hidden.gin * neti[i];
    }
  }
}

//}}}
//{{{  netResetGradientSums()

function netResetGradientSums() {

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    for (var h=0; h < netHiddenSize; h++) {
      output.gweightssum[h] = 0.0;
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    for (var i=0; i < netInputSize; i++) {
      hidden.gweightssum[i] = 0.0;
    }
  }
}

//}}}
//{{{  netAccumulateGradients()

function netAccumulateGradients() {

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    for (var h=0; h < netHiddenSize; h++) {
      output.gweightssum[h] += output.gweights[h];
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    for (var i=0; i < netInputSize; i++) {
      hidden.gweightssum[i] += hidden.gweights[i];
    }
  }
}

//}}}
//{{{  netApplyGradients()

function netApplyGradients(b,alpha) {

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    for (var h=0; h < netHiddenSize; h++) {
      output.weights[h] = output.weights[h] - alpha * (output.gweightssum[h] / b);
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    for (var i=0; i < netInputSize; i++) {
      hidden.weights[i] = hidden.weights[i] - alpha * (hidden.gweightssum[i] / b);
    }
  }
}

//}}}

neth[0].weights[0] = 6.0;
neth[0].weights[1] = -2.0;

neth[1].weights[0] = -3.0;
neth[1].weights[1] = 5.0;

neto[0].weights[0] = 1.0;
neto[0].weights[1] = 0.25;

neto[1].weights[0] = -2.0;
neto[1].weights[1] = 2.0;

//netInitWeights();

var i1 = [3,1];
var t1 = [1,0];

var i2 = [-1,4];
var t2 = [0,1];

var inputList  = [i1,i2];
var targetList = [t1,t2];

for (var epoch=0; epoch < 200; epoch++) {  // see page 33 - "iterate for 200 epochs"

  var loss = 0;

  netResetGradientSums();

  for (var i=0; i < inputList.length; i++) {

    var thisInput  = inputList[i];
    var thisTarget = targetList[i];

    netForward(thisInput)

    loss += netLoss(thisTarget);

    netCalcGradients(thisTarget);
    netAccumulateGradients();
  }

  //console.log ('loss',loss);

  netApplyGradients(1,0.5);  // a more usual learning rate is something like 0.001
}

console.log('outputs after 200 epochs, see page 33');

netForward(i1)
console.log(neto[0].out,neto[1].out);

netForward(i2)
console.log(neto[0].out,neto[1].out);

console.log('done');

