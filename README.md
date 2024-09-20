# Graph Cuts with Arbitrary Size Constraints Through Optimal Transport

This repository contains the code for the paper "Graph Cuts with Arbitrary Size Constraints Through Optimal Transport".



---


### Running the experiments

To run the experiments, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chakib401/OT-cut.git
   cd OT-cut
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Generate the image graphs**:
   ```bash
   mkdir graphs
   python generate_image_graphs.py
   ```

4. **Decompress the remaining graphs**:
    ```bash
    unzip graphs.zip -j graphs/
    ```

5. **Run the experiments**:
   ```bash
   python run_experiments.py
   ```

---

## Citation

If you find this work useful for your research, please consider citing us:

```bibtex
@article{
    anonymous2024graph,
    title={Graph Cuts with Arbitrary Size Constraints Through Optimal Transport},
    author={Anonymous},
    journal={Submitted to Transactions on Machine Learning Research},
    year={2024},
    url={https://openreview.net/forum?id=UG7rtrsuaT},
    note={Under review}
}
```

