//
// @colintomjenkins
//
// A simple Javascript implementation of @schienal's blog post titled
// "A worked example of backpropagation" at
// https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html
//
// Objects not used because of the longer term development intentions and planned
// optimisations inside Lozza chess https://github.com/op12no2/lozza
//
// Many of the loops can be collapsed, functions optimised and globals removed - but
// I've kept it all simple for clarity.
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

//
// Each node has 7 elements.
//

var NETIN          = 0;  // node input = sum of weights
var NETGIN         = 1;  // gradient of above
var NETOUT         = 2;  // node output = sigmoid(input)
var NETGOUT        = 3;  // gradient of above
var NETWEIGHTS     = 4;  // weights for node
var NETGWEIGHTS    = 5;  // gradients of above
var NETGWEIGHTSSUM = 6;  // sum of above when batching

var NETNODESIZE    = 7;

var neti = Array(netInputSize);
var neth = Array(netHiddenSize);
var neto = Array(netOutputSize);

for (var h=0; h < netHiddenSize; h++) {
  neth[h]                 = Array(NETNODESIZE);
  neth[h][NETIN]          = 0;
  neth[h][NETGIN]         = 0;
  neth[h][NETOUT]         = 0;
  neth[h][NETGOUT]        = 0;
  neth[h][NETWEIGHTS]     = Array(netInputSize);
  neth[h][NETGWEIGHTS]    = Array(netInputSize);
  neth[h][NETGWEIGHTSSUM] = Array(netInputSize);
}

for (var o=0; o < netOutputSize; o++) {
  neto[o]                 = Array(NETNODESIZE);
  neto[o][NETIN]          = 0;
  neto[o][NETGIN]         = 0;
  neto[o][NETOUT]         = 0;
  neto[o][NETGOUT]        = 0;
  neto[o][NETWEIGHTS]     = Array(netHiddenSize);
  neto[o][NETGWEIGHTS]    = Array(netHiddenSize);
  neto[o][NETGWEIGHTSSUM] = Array(netHiddenSize);
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
    x += (target[o] - neto[o][NETOUT]) * (target[o] - neto[o][NETOUT]);
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
    hidden[NETIN] = 0;
    for (var i=0; i < netInputSize; i++) {
      hidden[NETIN] += hidden[NETWEIGHTS][i] * neti[i];
    }
    hidden[NETOUT] = sigmoid(neth[h][NETIN]);
  }

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    output[NETIN] = 0;
    for (var h=0; h < netHiddenSize; h++) {
      output[NETIN] += output[NETWEIGHTS][h] * neth[h][NETOUT];
    }
    output[NETOUT] = sigmoid(output[NETIN]);
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
    output[NETGOUT] = 2 * (output[NETOUT] - targets[o]);
    output[NETGIN]  = dsigmoid(output[NETIN]) * output[NETGOUT];
  }

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    for (var h=0; h < netHiddenSize; h++) {
      var hidden = neth[h];
      output[NETGWEIGHTS][h] = output[NETGIN] * hidden[NETOUT];
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    hidden[NETGOUT] = 0;
    for (var o=0; o < netOutputSize; o++) {
      var output = neto[o];
      hidden[NETGOUT] += output[NETGIN] * output[NETWEIGHTS][h];
    }
    neth[h][NETGIN] = dsigmoid(neth[h][NETIN]) * neth[h][NETGOUT];
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    for (var i=0; i < netInputSize; i++) {
      hidden[NETGWEIGHTS][i] = hidden[NETGIN] * neti[i];
    }
  }
}

//}}}
//{{{  netResetGradientSums()

function netResetGradientSums() {

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    for (var h=0; h < netHiddenSize; h++) {
      output[NETGWEIGHTSSUM][h] = 0.0;
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    for (var i=0; i < netInputSize; i++) {
      hidden[NETGWEIGHTSSUM][i] = 0.0;
    }
  }
}

//}}}
//{{{  netAccumulateGradients()

function netAccumulateGradients() {

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    for (var h=0; h < netHiddenSize; h++) {
      output[NETGWEIGHTSSUM][h] += output[NETGWEIGHTS][h];
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    for (var i=0; i < netInputSize; i++) {
      hidden[NETGWEIGHTSSUM][i] += hidden[NETGWEIGHTS][i];
    }
  }
}

//}}}
//{{{  netApplyGradients()

function netApplyGradients(b,alpha) {

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    for (var h=0; h < netHiddenSize; h++) {
      output[NETWEIGHTS][h] = output[NETWEIGHTS][h] - alpha * (output[NETGWEIGHTSSUM][h] / b);
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    for (var i=0; i < netInputSize; i++) {
      hidden[NETWEIGHTS][i] = hidden[NETWEIGHTS][i] - alpha * (hidden[NETGWEIGHTSSUM][i] / b);
    }
  }
}

//}}}

neth[0][NETWEIGHTS][0] = 6.0;
neth[0][NETWEIGHTS][1] = -2.0;

neth[1][NETWEIGHTS][0] = -3.0;
neth[1][NETWEIGHTS][1] = 5.0;

neto[0][NETWEIGHTS][0] = 1.0;
neto[0][NETWEIGHTS][1] = 0.25;

neto[1][NETWEIGHTS][0] = -2.0;
neto[1][NETWEIGHTS][1] = 2.0;

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

  netApplyGradients(1,0.5);
}

console.log('outputs after 200 epochs, see page 33');

netForward(i1)
console.log(neto[0][NETOUT],neto[1][NETOUT]);

netForward(i2)
console.log(neto[0][NETOUT],neto[1][NETOUT]);

console.log('done');

