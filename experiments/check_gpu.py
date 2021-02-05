#!/usr/bin/env python3
import sys
sys.path.append('../') # or just install the module

if __name__== '__main__':
	### parser arguments
	import argparse

	parser = argparse.ArgumentParser('usage description')
	parser.add_argument('-gpu',  type=int, default=-1, help='gpu')
	main_args = parser.parse_args()

	###################################################################################################################################################
	import os
	if main_args.gpu>=0:
		os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # see issue #152
		os.environ['CUDA_VISIBLE_DEVICES'] = str(main_args.gpu) # CUDA-GPU

	###################################################################################################################################################
	import torch
	gpu_exists = torch.cuda.is_available()
	print(f'torch.cuda.is_available(): {gpu_exists}')
	if gpu_exists:
		device_name = torch.cuda.get_device_name(0)
		print(f'gpu_index: {main_args.gpu} - device_name: {device_name}')