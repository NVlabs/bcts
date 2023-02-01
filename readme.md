# Bellman Corrected Tree Search (BCTS)

This repository contains an implementation of the Bellman Corrected Tree Search (BCTS) algorithm, as described in the paper:

[Improve Agents without Retraining: Parallel Tree Search with Off-Policy Correction (NeurIPS 2021)]( https://proceedings.neurips.cc/paper/2021/file/2bd235c31c97855b7ef2dc8b414779af-Paper.pdf)

The BCTS algorithm is a search-based method for solving Reinforcement Learning building on [Rainbow](https://github.com/Kaixhin/Rainbow) [Hessel et al., 2017] and NVIDIA [CuLE](https://github.com/NVlabs/cule) [Dalton et al., 2019].

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker 19 or newer.
- Acess to NVIDIA Docker Catalog. Visit the [NGC website](https://ngc.nvidia.com/signup) and follow the instructions. This will grant you access to the base docker image (from the Dockerfile) and ability to run on NVIDIA GPU using the nvidia runtime flag.


### Installing

Clone the project to get the Dockerfile and build by running 
```
docker build -t bcts .
```

### Usage

1. Start the docker: 
   ```
   docker run --runtime=nvidia -it bcts /bin/bash
   ```
3. Run the code: 
   ```
   cd bcts; python main.py
   ``` 
   See main.py for optional parameters. For example, for tree depth 2 run: 
   ```
   python main.py --tree-depth=2
   ```

## License

This project is licensed under the NVIDIA License.

## Acknowledgments

If you use this project please cite:

@article{hallak2021improve,

  title={Improve agents without retraining: Parallel tree search with off-policy correction},
  
  author={Hallak, Assaf and Dalal, Gal and Dalton, Steven and Froisio, Iuro and Mannor, Shie and Chechik, Gal},
  
  journal={Advances in Neural Information Processing Systems},
  
  volume={34},
  
  pages={5518--5530},
  
  year={2021}
  
}




