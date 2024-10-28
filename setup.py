from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-E .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','')for req in requirements]

        if HYPEN_E_DOT in requrements:
            requ.remove(HYPEN_E_DOT)

    return requirements

setup(
    name = 'mlproject',
    version = '0.0.1',
    author='Kumar',
    autho_email = 'kmrstha9@gmail.com',
    packages = find_packages(),
    install_requires=get_requirements('requirements.txt')
)