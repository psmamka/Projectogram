# Projectogram and Reconstructogram

By: P. S. Mamkani

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


## Application Example

### Masked Reconstruction

In many imaging applications certain prior information about the object being imaged can be incorporated into the image reconstruction algorithm. One simple case is the non-negativity condition, i.e. image pixels are constrained to positive or zero values. This is especially the case when the relevant information is related to object density, as in X-ray where attenuation is related to the atomic number, or radioactive emission density, as in the cases of nuclear imaging techniques.

Masking is perhaps the simplest application of the projectogram, where a mask enclosing all possible nonzero image pixels is used to limit the reconstruction area. The following steps summarize the mask generatio process:

(1) Generate all single pixel projections --projectogram-- for the same projection angles as those used in the original image projection -- the sinogram
(2) Calculate the "support" of the image sinogram, i.e. the set of all projection elements with positive values
(3) For each pixel, if the support of the pixel projection is is a subset of the image projection support, the pixel belongs to the mask, and vice versa.

Once the reconstruction mask is generated, it is used to filter for pixels with positive values, while everything outside is set to zero.

Below we have the 50x50 image of the "Pacman" used as the reference -- or "phantom", -- the projections generated along 10 angles uniformly distributed from 0 degrees (inclusive) to 180 degrees (exclusive), and the reconstructed image using the pseudoinverse technique without applying any non--negativity constraints. The reason for using only the half-plane for selecting projection angles is that for "parallel beam" projections the data obtained from opposite angles are identical up to a reflection around the center. This is generally not the case for point-source/fan-beam geometries involving depth-dependent magnification effects.

![Pacman 50x50 - Sinogram - Pseudoinverse Recon](/figures/fig05_pacman50_proj_recon.png)
Figure 5. A 50x50 pacman image reconstructed from 10 projections without masking enforcement

By generating the mask and applying it to the reconstructed image, some of the spusrous reconstruction artifacts on the periphery can be removed:

![Pacman mask - masked recon](/figures/fig06_pacman50_masked_recon.png)
Figure 6. Enforcing of non-negativity condition and masking to arrive at the masked reconstruction

The streak artifacts within the centeral region of the image, contained inside the conves mask, are not removed, however. The masking technique is especially of significance when dealing with sparsely populated images.

Below we have the reference image of three small 2x2 squares within the 50x50 matrix, reconstructed using only 5 projection angles, with and without masking:

![Sparse image - psinv recon - masked recon](/figures/fig07_sparse50_masked_recon.png)
Figure 7. Reconstruction of a sparse image of three 2x2 squares before and after application of the mask

It is noted that presenting the reconstructogram for non-linear algorithms can be misleading, as single pixel reconstruction would be almost exact up to a normalization constant -- which is not the case for dense images.

## General Remarks

The projectogram and reconstructogram techniques are presented as an aid for qualitative visualiation of the projection and (linear) reconstruction operations, respectively. The rank of the projectogram provides essential information regarding the dimensionality of the projection data. The ratio of the projectogram rank to the number of indeendent basis vectors in the image space, i.e. number of pixels, provides us with insight concerning the sparseness of the information, and subsequently the size of the solution null space.

Use of prior information regarding possible image solutions can help up search for "better" solutions by limiting the size of the null space. One such assumption is the pixel non-negativity constraint, of relevance to most medical imaging modalities. The masked reconstruction presented here as a simple application of the projectogram technique along with the non-negativity constraint, whereby pixels are filtered based on the criterion of having projection footprints contained within the image projection support. It is argued that in cases of reconstruction of sparse data/images -- e.g. in imaging scenarios searching for subtle changes in metabolic signatures in confined areas, say metastases hotspots in the backdrop of stationary anato-physiological data -- the masked approach could be of significance.

## Selected Resources

For a quick introduction to tomography, see the Wikipedai page on tomograph reconstruction [here](https://en.wikipedia.org/wiki/Tomographic_reconstruction). For a classic introduction to Radon transform, see S. R. Dean's book [here](https://books.google.com/books?id=BXiXswEACAAJ). Kak and Slaney's book on compterized tomographic imaging is available [here](http://www.slaney.org/pct/).

The TomoPy library ([PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4181643/), [GitHub](https://github.com/tomopy/tomopy)) implements multiple techniques for image reconstruction in python. For cone-beam x-ray image reconstruction, CONRAD ([PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3820625/) [GitHub](https://github.com/akmaier/CONRAD)) is available. The ASTRA toolbox ([GitHub](https://github.com/astra-toolbox/astra-toolbox)) and the OSCaR package([UToronto](http://www.cs.toronto.edu/~nrezvani/OSCaR.html)) provide Matlab interface for their image reconstruction code. PYRO-NN ([GitHub](https://github.com/csyben/PYRO-NN)) provides a framework for integration of image reconstruction algorithms into the TensorFlow deep learning environment.


First Published:    13-Oct-2022
Last Updated:       14-Oct-2022

