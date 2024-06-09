# Sub Optimal Transport
A small python script generating a matlab animation of going through evenly spaced powers of an OT map.

## Motivation
The following technique was shown to me to recreate an image as well as possible by permuting (With an optimal transport map) the pixels of the initial image. https://nbviewer.org/github/gpeyre/numerical-tours/blob/master/matlab/optimaltransp_3_matching_1d.ipynb.
I was curious for the properties of the permutation generated by this technique.

- How many cycles would there be? Of what size?
- What would be the order?
- And my biggest motivation: Are there some visually interesting patterns that arise by going through the powers of such a permutation?

## Images
the Green Team Twente 'H' and the Hydriven Twente (the new brand of Green Team) 'H' logos. 
This seemed like a fitting example as the team has stayed the same with a different look, similar to how
the pixels of the image say the same but just rearrange.

## Method
The code is the documentation (sorry not sorry)


## Result and discussion
```
perm_lengths [2, 2, 3, 5, 27, 55, 102, 147, 174, 218, 465, 1469, 7331]
permutation degree 2610794299037414490
```
The resulting video first spends 1 second on the initial image, then runs through the powers of the form
n*([permutation degree]/465). This seemed to show some of the initial logo flickering through. 
This might be because we are essentially keeping all the other cycles the identity, 
and only cycling through the cycle with length 465.
They all stay the identity because they all divide the powers we use (by the way the degree is computed)

https://github.com/pixelplayer14/subOptimalTransport/assets/44682324/af1f2e83-0d24-4a96-86f2-e248dfb5c086

