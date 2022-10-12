# Projectogram and Reconstructogram
A projectogram is a visual representation of the operation of projecting higher dimenesional data to a lower dimensional space, whereby projections of single-pixel images are presented in a matrix form. A reconstructogram, on the other hand, visualizes the reconstruction outcome of single pixel images from the available projections.

## Motive
The concept of having mere access to a smaller, more compact form of data compared to the existing outer information is central to any form of physical measurements, and perhaps all human observations in general. And yet the issue of reduced dimensionality is of prominent significance in a special class of inverse problems, known as tomographic image reconstruction. Salient examples in this category include medical imaging modalities based on use of X-rays, radioactive molucules, and magnetic resonance phenomena.

In the continuous regime the forward and and inverse operations are characterizes by the Radon transform and its inverse. Under certainconditions, the analytical form of the Radon tranform inverse, filtered backprojection, is used for image reconstruction when projection data can be treated as a discrete sampling of the continuous "true" information. Yet in non-ideal settings when the discretization approximation is untenable, other classes of algebraic tecniques are used to achieve reconstruction.

It is noteworthy that while linearity assumption generally holds true in practice, with possible exceptions of digitization error and detector saturation, the "shift invariance" condition is typically violated due to myriad factors that affect image quality differently at the center versus the "edges" of the image matrix, where image quality tends to be lower. Therefore an exact "point spread function" can only be defined locally.

The technique presented here is less about solving a particular inverse problem of image reconstruction, rather focusing on a reprsentation of the qualities of projection and reconstruction systems. 

As a particular example, an image masking technique is presented where non-negative prior information about image pixel values is used to improve reconstructed image quality. 

## Technique

### Projectogram
A projectogram is a 2-dimensional representation of 4+ dimensional information, where all projections from all single-pixel images are represented in a single matrix form. This is achieved by "unraveling" of projection data for all angles, then "stacking" them horizontally or vertically side by side, where each row or column would represent projection information for a given sinpix image. Due to linearity of the projection operation, the projectogram can be viewed as the matrix representation of the projection operation in discrete domain, i.e. the discrete Radon transform, as a mapping from picture exelemts -- pixels -- to projection elements.

The rank of the projectogram determines existence of a unique solution for the system of linear equations.

Example of (a) a single pixel image in a 20x20 matrix, and (b) projections along 12 independent angles with 15 degree separation. Projection are on a detector array the with the same physical size and number of elements as rows of the image matrix.

Note that for some projection angles -- zero-based index from 2 to 4 -- the detector length is not sufficient to span the entire projection, resulting in cropped data. This is a contributing factor to the above mentioned non-shift-invariance property of the transformation.

![sigle pixel image and projections](/figures/fig01_single_pixel_image_projections.png)
Figure 1. A single-pixel image and its 12 projections

Due to the nearest-neighbor assignment of values from pixels to the delection elements, the projection values are binary.

The projectogram, which is a matrix representation of the forward projection operation, is obtained by creating single pixel projections for all pixels, followed by layering them in a matrix form where each line represents projection information obtained from a single pixel image.

Similar to the previous images, the values here are all binary as well.

![projectogram 20x20 12 angs](/figures/fig02_projectogram_20x20.png)
Figure 2. The projectogram generated from the 12 projection angles

### Reconstructogram

On the other side of the coin from the projectogram is the reconstructogram, or simply recongram. Here, instead of displaying single-pixel image projections we are displaying the result of the recunstruction algorithm applied to sinpix projections, and similarly displaying the 4-or-more dimensional results in a 2d matrix, where each row represents the linearized form the sinpix reconstruction result. A recongram would be identical to the identity matrix for an ideal reconstruction.

A conceptually simple way of applying an approximate inverse transform when the projectogram rank is less than the number of degrees of freedom, i.e. number of pixels, is to use the Moore-Penrose (pseudo-)inverse. In simple terms, the pseudoinverse method finds the least-squares fit solution by inverting the non-zero elements of the singular-value-decompostion diagonal matrix while keeping the rest at zero.

For the previous example of a non-full rank projection of 20-by-20 matrix along 12 angles (15 degree separation covering the half-plane), we have the Moore-Penrose pseudoinverse calculated using the implementation from the scipy library. Note that the pseudoinverse rank (239) is slightly less than the number of projection elements of 240, i.e. angles (12) times the detector array length (20).

![Psuedoinverse plot](/figures/fig03_pseudoinverse_20x20.png)
Figure 3. Moore-Penrose pseudoinverse

Using the above pseudoinverse, the reconstruction of the sinpix (2,3) is achieved as shown below on the left and the full reconstructogram on the right. The streak artifacts in the single-pixel reconstruction appear as non-diagonal elements in the recongram.

![Recongram 20x20 12 angles](/figures/fig04_sinpix_recongram_20x20.png)
Figure 4. Single pixel reconstruction at (2,2) location, representing a single row in the full reconstructogram

