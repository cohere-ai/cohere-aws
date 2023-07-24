import setuptools
from setuptools.command.install import install
from setuptools.dist import Distribution

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


class InstallPlatlib(install):

    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):

    def is_pure(self) -> bool:
        return False

    def has_ext_modules(foo) -> bool:
        return True


setuptools.setup(name='cohere-sagemaker',
                 version='0.7.2',
                 author='Cohere',
                 author_email='support@cohere.ai',
                 description='A Python library for the Cohere endpoints in AWS Sagemaker',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/cohere-ai/cohere-sagemaker',
                 packages=setuptools.find_packages(),
                 install_requires=['boto3', 'sagemaker'],
                 include_package_data=True,
                 classifiers=[
                     'Programming Language :: Python :: 3',
                     'License :: OSI Approved :: MIT License',
                     'Operating System :: OS Independent',
                 ],
                 python_requires='>=3.6',
                 distclass=BinaryDistribution,
                 cmdclass={'install': InstallPlatlib})
