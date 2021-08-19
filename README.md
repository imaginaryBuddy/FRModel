# FRModel

Forest Recovery Model Research Project.

This project is under the **Nanyang Technological University** (NTU) **Undergraduate Research Experience on Campus** (URECA) Program.

# Install

**It is recommended to fork for the latest updates**. The conda package will not be maintained.

To build the Cython files, you need to run the one-liner `c_setup.bat`.

Follow the following installation steps if not on Windows
```bash
$ python c_setup.py build_ext --inplace
$ cd src && pip install .
```

Test the installation by running the following command in your Python shell
```
import frmodel
```

No errors should pop up.

# Dependencies

The following repository requires `GDAL`. Refer to the following website for more information for the installation of `GDAL`: https://gdal.org/download.html

For Mac OS > 10, you may run `brew install GDAL` to install the package.

## Python Packages

Run the following script in the main directory to install the required Python packages

```bash
pip install -r requirements.txt
```

# License

Licensed under the **Mozilla Public License 2.0**

https://github.com/Eve-ning/FRModel/blob/master/LICENSE

# Citation

If you have used or referenced any of the code in the repository,
please kindly cite

```
@misc{frmodel,
  author = {John Chang},
  title = {FRModel: Forest Recovery Modelling},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Eve-ning/FRModel}},
}
```
