# IPBLoaders

Collection of commonly used dataloaders for python.<br />
For any doubts contact Federico, or Matteo.

# Installation

`pip3 install -U -e .`

Note that the depencies are not installed by default. There's however a requirements file for now.

# How to add a Dataloader

Let's say that `Pierugo` is dataloader class loading an `rgb` dataset.<br />
If you want to add `Pierugo` into this library, here are the steps you need to do:

- Copy the file containing `Pierugo` into the corresponding folder. For now we keep a per-sensor organization.
- Make `Pierugo` a child of `ipb_loaders.ipb_base.IPB_BASE`.
- Set `Pierugo.data_source = <path-on-your-machine>`.
- Run `python scripts/single_test.py -t rgb -d Pierugo` and solve all errors.
- Set `Pierugo.data_source = None`.
- Push.

# How to use a Dataloader

Here's a snippet to use `Pierugo` in your implementations

```
from ipb_loaders.rgb.pierugo import Pierugo
from torch.utils.data import DataLoader
dataloader = DataLoader(Pierugo(), collate_fn=Pierugo.collate)
for item in dataloader:
    <do stuff>

```

# TODOs before going public

- [x] download data if data_source not specified
- [x] protect master branch
- [ ] push creates a new branch, run tests and merge if nothing is failing
- [x] add more datasets
- [x] add more tests
- [x] add lidar datasets
- [ ] add lidar-specific tests
- [x] find a way to remove test.utils.imports
- [x] Base test class
- [ ] pip installable
- [x] formatter file
- [x] argument for local testing single dataloaders
- [x] default download folder if data_source is not specified
- [ ] download and extract if zip
- [ ] poses tests
- [ ] check if both `setup.py` and `install.py` are required

# Discussion points

- what do we do for the dependecies?
- collate for each dataset?
- handle imports in the `__init__` or not?
- cache?