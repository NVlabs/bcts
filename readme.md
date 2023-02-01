# Bellman Corrected Tree Search (BCTS)

This repository contains an implementation of the Bellman Corrected Tree Search (BCTS) algorithm, as described in the paper:

[Improve Agents without Retraining: Parallel Tree Search with Off-Policy Correction (NeurIPS 2021)]( https://proceedings.neurips.cc/paper/2021/file/2bd235c31c97855b7ef2dc8b414779af-Paper.pdf)

The BCTS algorithm is a search-based method for solving Reinforcement Learning problem building on [Rainbow](https://github.com/Kaixhin/Rainbow) implementation. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker
- Create an NGC account by visiting the [NGC website](https://ngc.nvidia.com/signup) and following the instructions.


### Installing

Download the "Dockerfile" file from the repository in a new folder. 

Run `docker build -t bcts .`

### Usage

You can use the `bcts.py` module in your own project or run the example script `example.py` to see the BCTS algorithm in action.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

We urge anyone using this work (code or paper) to cite:

@article{dalal2021improve,

  title={Improve agents without retraining: Parallel tree search with off-policy correction},
  
  author={Dalal, Gal and Hallak, Assaf and Dalton, Steven and Mannor, Shie and Chechik, Gal and others},
  
  journal={Advances in Neural Information Processing Systems},
  
  volume={34},
  
  pages={5518--5530},
  
  year={2021}
  
}




