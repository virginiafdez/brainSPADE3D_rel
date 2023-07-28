# brainSPADE3D_rel

This is the code that was used for: "A 3D generative model of pathological multi-modal MR images and segmentations" by Virginia Fernandez, Walter Hugo Lopez Pinaya, Pedro Borges, Mark Graham, Tom Vercauteren and M Jorge Cardoso. Reference will be inserted upon its availability. 
The codebase is based on
- brainSPADE2D (https://github.com/virginiafdez/brainSPADE_RELEASE.git)
- NVIDIA SPADE (https://github.com/NVlabs/SPADE.git)
and has been modified to provide label generation + semantic image synthesis code in 3D.

To use the code, you first need to use git clone https://github.com/Project-MONAI/GenerativeModels.git into the repo, as it is used by the label generator functions. 

Runnable scripts:
- train_autoencoder.py : trains the Spatial VAE used for the label generator
- train_ldm.py : trains the diffusion model used for the label generator
- train.py : trains the SPADE3D network

These files can be used in run_scripts.
The first 2 use yaml configuration files. Examples can be found in saved_configs. The configurations that are present there ARE NOT the ones from the paper. The paper configurations can be found in its Supplementary Materials. 
