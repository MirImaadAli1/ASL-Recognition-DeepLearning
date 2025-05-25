# CorrNet-ASL

This repository adapts the **CorrNet** architecture from the CVPR 2023 paper [Continuous Sign Language Recognition with Correlation Network](https://arxiv.org/abs/2303.03202) for **American Sign Language (ASL)** using the **How2Sign** dataset. The goal of this project is to perform continuous sign language recognition (CSLR) by leveraging both temporal correlations and spatial features for improved accuracy.

## About How2Sign Dataset

[How2Sign](https://how2sign.github.io/) is a large-scale, continuous **American Sign Language (ASL)** dataset that contains **multi-view RGB videos**, **depth videos**, and **3D skeletal data**. It includes diverse everyday sentences signed by native ASL signers. The dataset is designed for tasks such as continuous sign language recognition and translation. 

For this project, we focus on **front-view RGB videos** from the How2Sign dataset, using them for training and evaluation of our model.

## Key Features of the Project

- **Adaptation of CorrNet**: This repository modifies the **CorrNet** model to work with the **How2Sign** dataset, focusing on continuous ASL recognition.
- **Greedy Search Decoding**: Initially, beam search decoding via `ctcdecode` was considered, but due to high computational requirements and limited support on non-Linux systems, we switched to **greedy decoding** for improved efficiency.
- **Preprocessing Pipeline**: A preprocessing pipeline is included to handle How2Sign’s data format (multi-view RGB videos), making it easier to set up the dataset for training and inference.
- **Evaluation Metrics**: We use **Word Error Rate (WER)** as the primary evaluation metric to assess model performance.

## Installation

To get started, clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/CorrNet-ASL.git
cd CorrNet-ASL
pip install -r requirements.txt
```

Ensure that you have the required hardware (preferably a CUDA-enabled GPU) for training.

## Dataset Preparation

1. **Download How2Sign Dataset**:
   - The dataset can be found on the [How2Sign website](https://ict.usc.edu/how2sign/).
   - You’ll need to register and request access to the data. After downloading, organize the dataset so that it contains the necessary **front-view RGB videos**.

2. **Preprocessing**:
   - The preprocessing steps will convert the raw videos into frames and create a file structure for training. You can use the provided preprocessing script to prepare the data.
   
   Example command for preprocessing:

   ```bash
   python preprocess.py --dataset_path /path/to/how2sign --output_dir ./processed_data
   ```

3. **Gloss Dictionary**:
   - A gloss dictionary is required to map the sentences to their corresponding labels. You can find an example glossary in the dataset or create your own if necessary.

## Training the Model

After preparing the dataset, you can begin training the model. The configuration file (`./config/baseline.yaml`) contains all necessary training parameters (e.g., learning rate, batch size, etc.).

To train the model, run:

```bash
python main.py --config ./config/baseline.yaml --device cuda
```

If you need to modify training parameters (such as dataset path, gloss dictionary, etc.), you can edit the `baseline.yaml` file.

## Inference and Evaluation

After training, you can test your model with the following command:

```bash
python main.py --config ./config/baseline.yaml --device cuda --load-weights path_to_model.pt --phase test
```

This will load the trained model from `path_to_model.pt` and evaluate it on the test set, reporting the **Word Error Rate (WER)**.

### Example Command:

```bash
python main.py --config ./config/baseline.yaml --device cuda --load-weights ./models/corrnet_asl.pt --phase test
```

## Performance

The model is evaluated using **Word Error Rate (WER)**, which is the standard metric for ASL recognition tasks. Training results, including WER, will be logged to the output for each epoch.

## Acknowledgements

This project is built on the **CorrNet** architecture, originally designed for **continuous sign language recognition**. Special thanks to the authors for providing the original codebase, which was modified for the How2Sign dataset in this repository. 

We also acknowledge the **How2Sign** team for making their dataset publicly available for research purposes.

## Citation

If you use this repository or parts of this work in your research, please cite the following papers:

- **CorrNet**: 
  - Y. Li, Y. Liu, Z. Xie, H. Li, M. Xu, and S. Wang, “Continuous Sign Language Recognition with Correlation Network,” CVPR 2023.
- **How2Sign**: 
  - T. Z. P. Hamid, Y. H. Kim, et al., “How2Sign: Large-Scale Continuous American Sign Language Dataset,” in Proceedings of the ICCV 2021.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
