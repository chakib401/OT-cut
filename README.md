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


