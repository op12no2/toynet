# toynet
A Javascript version of Alexander Schiendorfer's blog post "A worked example of backpropagation".

https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html

Download the toynet.js and run in a Javascript interpreter like Nodejs for example. By default it runs the batching example on page 33, but
it's easy to play around with the model using console.log().

The code is generalised a bit so the shape of the network can be changed. I added bias and alternative activatiion functions. 

Alexander's post was the one that made everything crystal clear to me; it's worth working through page by page, using the Javascript model to console.log() the relevant data as it's presented in the post.
