#!/usr/bin/python
import argparse

def reduce_deformation_fields(file_name,output_name):
	import nibabel
	five_d_file = nibabel.load(file_name)
	data = five_d_file.get_data()
	four_d_file=nibabel.Nifti1Image(data[:,:,:,0,:], five_d_file.get_affine())
	nibabel.save(four_d_file, output_name)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='reducing dimensions of ANTs specific 5D deformation fields')
    parser.add_argument("-i", dest="file_name",help="path to 5d deformation field", required=True)
    parser.add_argument("-o", dest="output_name",help="4d output file name",required=True)
    args = parser.parse_args()
    ret = reduce_deformation_fields(args.file_name, args.output_name)

