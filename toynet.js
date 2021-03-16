//
// @colintomjenkins
//
// A simple Javascript implementation of @schienal's blog post titled
// "A worked example of backpropagation" at
// https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html
//
// Objects not used because of the longer term development intentions and optimisations
// inside Lozza chess https://github.com/op12no2/lozza
//
// Also many of the loops can be collapsed, functions optimised and globals removed - but
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

//
// I've also added bias and alternative activators on a per layer basis.
//

//{{{  build net

var NETIN          = 0;  // node bias + sum of weights
var NETGIN         = 1;  // gradient of above
var NETOUT         = 2;  // result of activation applied to NETIN
var NETGOUT        = 3;  // gradient of above
var NETWEIGHTS     = 4;  // weights for a node
var NETGWEIGHTS    = 5;  // gradients of above
var NETGWEIGHTSSUM = 6;  // sum of above when batching
var NETBIAS        = 7;  // bias for a node
var NETGBIAS       = 8;  // gradient of above
var NETGBIASSUM    = 9;  // sum of above when batching

var NETCELLSIZE    = 10;  

var neti = Array(netInputSize);
var neth = Array(netHiddenSize);
var neto = Array(netOutputSize);

for (var h=0; h < netHiddenSize; h++) {
  neth[h]                 = Array(NETCELLSIZE);
  neth[h][NETIN]          = 0;
  neth[h][NETGIN]         = 0;
  neth[h][NETOUT]         = 0;
  neth[h][NETGOUT]        = 0;
  neth[h][NETBIAS]        = 0;
  neth[h][NETGBIAS]       = 0;
  neth[h][NETGBIASSUM]    = 0;
  neth[h][NETWEIGHTS]     = new Float64Array(netInputSize);
  neth[h][NETGWEIGHTS]    = new Float64Array(netInputSize);
  neth[h][NETGWEIGHTSSUM] = new Float64Array(netInputSize);
}

for (var o=0; o < netOutputSize; o++) {
  neto[o]                 = Array(NETCELLSIZE);
  neto[o][NETIN]          = 0;
  neto[o][NETGIN]         = 0;
  neto[o][NETOUT]         = 0;
  neto[o][NETGOUT]        = 0;
  neto[o][NETBIAS]        = 0;
  neto[o][NETGBIAS]       = 0;
  neto[o][NETGBIASSUM]    = 0;
  neto[o][NETWEIGHTS]     = new Float64Array(netInputSize);
  neto[o][NETGWEIGHTS]    = new Float64Array(netInputSize);
  neto[o][NETGWEIGHTSSUM] = new Float64Array(netInputSize);
}

//}}}

//{{{  activators

var netOutputActivator = [];
var netHiddenActivator = [];

function sigmoid(x) {
  return (1.0 / (1.0 + Math.exp(-x)));
}

function dsigmoid(x) {
  return sigmoid(x) * (1.0 - sigmoid(x));
}

function relu(x) {
  if (x > 0)
    return x;
  else
    return 0;
}

function drelu(x) {
  if (x > 0)
    return 1;
  else
    return 0;
}

function linear(x) {
  return x;
}

function dlinear(x) {
  return 1;
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
    hidden[NETIN] = hidden[NETBIAS];
    for (var i=0; i < netInputSize; i++) {
      hidden[NETIN] += hidden[NETWEIGHTS][i] * neti[i];
    }
    hidden[NETOUT] = netHiddenActivator[0](neth[h][NETIN]);
  }

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    output[NETIN] = output[NETBIAS];
    for (var h=0; h < netHiddenSize; h++) {
      output[NETIN] += output[NETWEIGHTS][h] * neth[h][NETOUT];
    }
    output[NETOUT] = netOutputActivator[0](neto[o][NETIN]);
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
    output[NETGIN]  = netOutputActivator[1](output[NETIN]) * output[NETGOUT];
  }

  for (var o=0; o < netOutputSize; o++) {
    var output = neto[o];
    output[NETGBIAS][h] = output[NETGIN] * 1;
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
    neth[h][NETGIN] = netHiddenActivator[1](neth[h][NETIN]) * neth[h][NETGOUT];
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    hidden[NETGBIAS] = hidden[NETGIN] * 1;
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
    output[NETGBIASSUM] = 0.0;
    for (var h=0; h < netHiddenSize; h++) {
      output[NETGWEIGHTSSUM][h] = 0.0;
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    hidden[NETGBIASSUM] = 0.0;
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
    output[NETGBIASSUM] += output[NETGBIAS];
    for (var h=0; h < netHiddenSize; h++) {
      output[NETGWEIGHTSSUM][h] += output[NETGWEIGHTS][h];
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    hidden[NETGBIASSUM] += hidden[NETGBIAS];
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
    output[NETBIAS] = alpha * (output[NETGBIASSUM] / b);
    for (var h=0; h < netHiddenSize; h++) {
      output[NETWEIGHTS][h] = output[NETWEIGHTS][h] - alpha * (output[NETGWEIGHTSSUM][h] / b);
    }
  }

  for (var h=0; h < netHiddenSize; h++) {
    var hidden = neth[h];
    hidden[NETBIAS] = alpha * (hidden[NETGBIASSUM] / b);
    for (var i=0; i < netInputSize; i++) {
      hidden[NETWEIGHTS][i] = hidden[NETWEIGHTS][i] - alpha * (hidden[NETGWEIGHTSSUM][i] / b);
    }
  }
}

//}}}

neth[0][NETBIAS]       = 0.0;
neth[0][NETWEIGHTS][0] = 6.0;
neth[0][NETWEIGHTS][1] = -2.0;

neth[1][NETBIAS]       = 0.0;
neth[1][NETWEIGHTS][0] = -3.0;
neth[1][NETWEIGHTS][1] = 5.0;

neto[0][NETBIAS]       = 0.0;
neto[0][NETWEIGHTS][0] = 1.0;
neto[0][NETWEIGHTS][1] = 0.25;

neto[1][NETBIAS]       = 0.0;
neto[1][NETWEIGHTS][0] = -2.0;
neto[1][NETWEIGHTS][1] = 2.0;

var i1 = [3,1];
var t1 = [1,0];

var i2 = [-1,4];
var t2 = [0,1];

netHiddenActivator = [sigmoid,dsigmoid];
//netHiddenActivator = [relu,drelu];
netOutputActivator = [sigmoid,dsigmoid];

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

netForward(i1)
console.log(neto[0][NETOUT],neto[1][NETOUT]);

netForward(i2)
console.log(neto[0][NETOUT],neto[1][NETOUT]);

console.log('done');

