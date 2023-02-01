# Bellman Corrected Tree Search (BCTS)

This repository contains an implementation of the Bellman Corrected Tree Search (BCTS) algorithm, as described in the paper:

https://proceedings.neurips.cc/paper/2021/file/2bd235c31c97855b7ef2dc8b414779af-Paper.pdf

The BCTS algorithm is a search-based method for solving Reinforcement Learning problem building on Rainbow (https://github.com/Kaixhin/Rainbow) implementation. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker
- Create an NGC account by visiting the [NGC website](https://ngc.nvidia.com/signup) and following the instructions.


### Installing

Download the Dockerfile file from the repository in a new folder. 

Run `docker build -t bcts .`

### Usage

You can use the `bcts.py` module in your own project or run the example script `example.py` to see the BCTS algorithm in action.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- The authors of the original research paper on BCTS.
- The developers and maintainers of the packages used in this implementation.





