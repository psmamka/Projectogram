# Projectogram and Reconstructogram
A projectogram is a visual representation of the operation of projecting higher dimenesional data to a lower dimensional space, whereby projections of single-pixel images are presented in a matrix form. A reconstructogram, on the other hand, visualizes the reconstruction outcome of single pixel images from the available projections.

## Motive
The concept of having mere access to a smaller, more compact form of data compared to the existing outer information is central to any form of physical measurements, and perhaps all human observations in general. And yet the issue of reduced dimensionality is of prominent significance in a special class of inverse problems, known as tomographic image reconstruction. Salient examples in this category include medical imaging modalities based on use of X-rays, radioactive molucules, and magnetic resonance phenomena.

In the continuous regime the forward and and inverse operations are characterizes by the Radon transform and its inverse. Under certainconditions, the analytical form of the Radon tranform inverse, filtered backprojection, is used for image reconstruction when projection data can be treated as a discrete sampling of the continuous "true" information. Yet in non-ideal settings when the discretization approximation is untenable, other classes of algebraic tecniques are used to achieve reconstruction.

It is noteworthy that while linearity assumption generally holds true in practice, with possible exceptions of digitization error and detector saturation, the "shift invariance" condition is typically violated due to myriad factors that affect image quality differently at the center versus the "edges" of the image matrix, where image quality tends to be lower. Therefore an exact "point spread function" can only be defined locally.

The technique presented here is less about solving a particular inverse problem of image reconstruction, rather focusing on a reprsentation of the qualities of projection and reconstruction systems. 

As a particular example, an image masking technique is presented where non-negative prior information about image pixel values is used to improve reconstructed image quality. 

## Technique
A projectogram is a 2-dimensional representation of 4+ dimensional information, where all projections from all single-pixel images are represented in a single matrix form. This is achieved by "unraveling" of projection data for all angles, then "stacking" them horizontally or vertically side by side, where each row or column would represent projection information for a given sinpix image. Due to linearity of the projection operation, the projectogram can be viewed as the matrix representation of the projection operation in discrete domain, i.e. the discrete Radon transform, as a mapping from picture exelemts -- pixels -- to projection elements.

The rank of the projectogram determines existence of a unique solution for the system of linear equations.

Example of (a) a single pixel image in a 20x20 matrix, and (b) projections along 12 independent angles with 15 degree separation. Projection are on a detector array the with the same physical size and number of elements as rows of the image matrix.

Note that for some projection angles -- zero-based index from 2 to 4 -- the detector length is not sufficient to span the entire projection, resulting in cropping. This contributes to the above mentioned non-shift-invariance quality of the transformation.
