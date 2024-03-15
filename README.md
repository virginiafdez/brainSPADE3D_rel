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

## Format of the data

This section explains the requirements that the data has to fulfill to run the code, and the ones that are desirable. 
For both the label generator training and the image generator, the data is provided via a TSV file:
- For the label generator: a column "label" should point to the nii.gz file containing the 3D label. These labels should have [X,Y,Z,C] shape, where X,Y,Z are spatial dimensions, and C is the number of semantic channels. If lesion conditioning is used, the name of the lesion conditioning passed to the diffusion model "e.g. "tumour"" should match the column header. Each row should have a float value indicating the proportion of lesion present in each specific level (0.000 meaning "not present"). To ensure stability, users are recommended to normalise this proportion, ensuring that no conditioning value exceeds 1.
- For the image generator, a column "label" should point to the nii.gz file containing the 3D label. Subsequent columns should have corresponding modality images (e.g. "T1", "T2" etc.). These modality names should match those passed to the image generator under argument "sequences". There can be missing modalities: leave as "none", but there has to be AT LEAST one image per label. The labels should have the same format as indicated before. The images should have [X,Y,Z] shape, where X,Y,Z are spatial dimensions.

How to create these TSV files? You can write your own code. The trainers should work as long as the TSVs satisfy the requirements above. Nonetheless, there are functions you can use in data>dataset_utils.py:
- create_datatext_source_image_gen: which creates the image generator TSV files, if you pass dictionaries with label and modality folders (see input arguments to the function for more details).
- create_datatext_source_label_gen: which creates the label generator TSV files. This function depends on the previous, as it takes the TSV file of the previous function. This is to ensure that the train, val, test splits are exactly the same. This function also loops through the label list, loads each label, calculates the disease proportion (number of voxels in the image) for each of the lesions passed as an argument, then normalises the proportion by the biggest one across the dataset.
- rewrite_tsv_with_new_sources: this function can come handy if you need to rewrite the paths within a TSV file to something else (relative path changes). 
