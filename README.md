## DiffAnnot: Improved Neural Annotator with Denoising Diffusion Model

Authors: [Chaofan Lin](https://github.com/SiriusNEO), [Tianyuan Qiu](https://github.com/PaperL), [Hanchong Yan](https://github.com/brandon-yan), [Muzi Tao](https://github.com/Seanzzia)

- [Source Code](https://github.com/PaperL/Human-3D-Diffusion)

- [Paper In PDF](DiffAnnot.pdf)

This project uses [MMHuman3D](https://github.com/open-mmlab/mmhuman3d/).

### Abstract

3D human reconstruction is an important task in computer vision to generate human 3D models from photos or videos. But the research of reconstruction requires human body data with annotation.  Previous widely used neural-network-based annotators annotate 3D human models automatically but there is still room for quality improvement. In our research, we innovatively regard the deviation between the annotations and true human poses as a type of noise. We propose a new annotator DiffAnnot using a denoising diffusion model to further refine the annotations from a pre-trained annotator. We train and test DiffAnnot on various datasets including 3DPM, MPI-INF-3DHP and Halpe. DiffAnnot is evaluated on several widely used metrics including MPJPE and shows outstanding performance.

### Architecture

![](pipeline.png)


### Examples of Annotations
<img style="margin: 0px 40px;" src="mesh.jpg" width="2000"/>
<br>
Above: HMR Results. 　Below: Our Results.