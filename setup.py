from setuptools import find_packages, setup


if __name__ == '__main__':
    setup(
        name='WHAM',
        version='0.1.0',
        description='OpenMMLab Pose Estimation Toolbox and Benchmark.',
        author='',
        author_email='',
        keywords='computer vision',
        packages=find_packages(exclude=('configs', 'tools')),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
        ],
        url='https://github.com/yohanshin/WHAM',
        license='Apache License 2.0',
        python_requires='>=3.8',
        zip_safe=False)