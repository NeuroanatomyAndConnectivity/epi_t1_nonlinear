epi_t1_nonlinear.py
================

Perfoms nonlinear EPI to T1 registration using antsRegistration. 
Before, the T1 image has to be processed in freesurfer and the EPI timeseries should be motion corrected.
The code works as a reusable nipype workflow or commandline tool. 

###nipype
example
```
nipype_epi_t1_nonlin = create_epi_t1_nonlinear_pipeline('nipype_epi_t1_nonlin')
nipype_epi_t1_nonlin.inputs.inputnode.fs_subject_id = '123456'
nipype_epi_t1_nonlin.inputs.inputnode.fs_subjects_dir = '/project/data/freesurfer'
nipype_epi_t1_nonlin.inputs.inputnode.realigned_epi = 'mcflirt.nii.gz'
nipype_epi_t1_nonlin.run()
```

inputs
```
inputnode.fs_subject_id    # subject id used in freesurfer
inputnode.fs_subjects_dir  # path to freesurfer output
inputnode.realigned_epi    # realigned EPI timeseries
```

outputs
```
outputnode.lin_epi2anat     # ITK format
outputnode.nonlin_epi2anat  # ANTs specific 5D deformation field
outputnode.nonlin_anat2epi  # ANTs specific 5D deformation field
```


###commandline

```
-h, --help    show this help message and exit
-epi EPI      realigned EPI timeseries
-fsdir FSDIR  path to freesurfer subjects directory
-fsid FSID    subject id used in freesurfer
-wd WD        working directory to store output
```





reduce_deformation_fields.py
================

Reduces dimensions of ANTs specific 5D deformation fields to 4
```
-h, --help      show this help message and exit
-i FILE_NAME    path to 5D deformation field
-o OUTPUT_NAME  4D output file name
```

