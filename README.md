# toynet
A Javascript version of Alexander Schiendorfer's blog post "A worked example of backpropagation".

https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html

Download the toynet.js and run in a Javascript interpreter like Nodejs for example.

The code is generalised a bit so the shape of the network can be changed.

Alexander's post was the one that made everything crystal clear to me; it's worth working through page by page, using the Javascript model to console.log() the relevant data as it's presented in the post.

If there are errors in the bias and relu/linear activator code, it's my fault - those are the bits I added :)

By default it runs the batching example on (A4 size) page 33.

```
[C:\projects\toynet\trunk]node toynet.js                                            
outputs after 200 epochs, see page 33                                                                                                          
0.9460731161243365 0.04903030671319556                                                                                                         
0.056063676671722545 0.9508238102520352                                                                                                        
done  
```
