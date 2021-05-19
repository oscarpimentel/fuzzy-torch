import setuptools

with open('README.md', 'r') as fh:
	long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
	name='fuzzy-torch',
	version='0.0.1',
	author='Óscar Pimentel Fuentes',
	author_email='oscarlo.pimentel@gmail.com',
	description='Library with personal pytorch utils',
	long_description=long_description,
    long_description_content_type='text/markdown',
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    #packages=['fuzzytools'],
	#license='MIT licence',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
	#keywords='times series features, light curves',
	#include_package_data=True,
	python_requires='>=3.7',
	install_requires=required,
)
