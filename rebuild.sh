rm -rf clddp.egg-info
rm -rf dist
rm -rf build
python setup.py sdist bdist_wheel
# upload: python -m twine upload dist/*
