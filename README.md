# Efficient and Interpretable Grammatical Error Correction with Mixture of Experts

This repository provides the code to train and test the MoE-based Grammatical Error Correction model from the following publication:

> Efficient and Interpretable Grammatical Error Correction with Mixture of Experts  
> [Muhammad Reza Qorib](https://mrqorib.github.io/), [Alham Fikri Aji](https://afaji.github.io/), and [Hwee Tou Ng](https://www.comp.nus.edu.sg/~nght/)  
> Findings of the 2023 Conference on Empirical Methods in Natural Language Processing ([PDF](https://mrqorib.github.io/assets/pdf/MoECE.pdf))

The codebase is built by modifying Fairseq [v0.9.0](https://github.com/facebookresearch/fairseq/tree/v0.9.0) (with the T5 model script from [Applica](https://github.com/applicaai/fairseq/tree/applica-t5)) and [FastMoE v1.1.0](https://github.com/laekov/fastmoe). Install the dependencies by running:

## Installation

Install the code dependencies by running the commands below:

```bash
# Install PyTorch according to your CUDA version, check https://pytorch.org/get-started/previous-versions/
pip install -e .
cd fastmoe/; USE_NCCL=0 python setup.py install; cd ..
pip install "numpy<1.24"
```

Download the pre-trained models using the command below:

```bash
git clone https://huggingface.co/mrqorib/MoECE models
```

## Inference

The input file needs to be tokenized using the T5 tokenizer. Standard test files are available under the `data/tokenized` directory in this repository. For a custom test set, tokenize the text by running the following command:

```bash
python scripts/hf-encode.py --model_name --input ${input_path} --output ${output_path}
```

Run the inference using the following command:

```bash
./moe-test.sh data/tokenized/${input_file} models/${model_file} ${output_path}
```

For evaluating the model on the CoNLL-2014 test set, retokenize the output using the script below:

```bash
python scripts/retokenize.py ${conll_output} > ${tokenized_conll_output}
```

## Re-training the model

If you want to reproduce the research or train the model with your own data, additional libraries are needed to prepare the training data. You may install the following libraries in a different virtual environment if you experience package conflicts.

Below is the list of libraries used during the research. Using a different version may work, but it could produce different results. ERRANT with different versions may generate different error types. Version 2.0.0 was used as it was the official version for the BEA-2019 shared task.

```
transformers==4.12.5
errant==2.0.0
```

### Prepare the training data

Please follow the instructions in the official [cLang-8 repository](https://github.com/google-research-datasets/clang8) to download the cLang-8 data.

Afterward, get the error types of the training data by running the following command:

```bash
python scripts/get_error_types.py --source_text ${source_path} --target_text ${target_path} --output_path ${error_labels_path}
```

Tokenize both the source and target texts using the tokenizer of the T5 model. When tokenizing the target text, please also provide the error labels so that the token indices match.

```bash
python scripts/hf-encode.py --model_name --input ${source_path} --output data/tokenized/clang8.ori
python scripts/hf-encode.py --model_name --input ${target_path} --aux_inputs ${error_labels_path} --output data/tokenized/clang8.cor --aux_outputs data/tokenized/clang8.edit
```

Binarize the training data according to the Fairseq format:

```bash
cd data
./scripts/edit-fairseq-preprocess.sh tokenized/clang8 fairseq-aux-bin
```

### Download T5-v1.1 weights

Download the T5 v1.1 weights from the [T5 official repository](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md).

Convert the weight format into the Fairseq format:

```bash
python tf2fairseq.py --arch t5-v1.1-${size} --tf-checkpoint {path_to_downloaded_checkpoints} --output models/pretrained/t5-v1.1-${size}
```

### Train the model

Run the following command to train the model. Provide `gs` as the `${moe_type}` argument to train a MoECE-GS model and `st` to train a MoECE-ST model. Provide the `${size}` argument with `base` or `large`.

```bash
./moe-train.sh ${moe_type} ${size}
```

After training is finished, choose the best checkpoint based on the highest F0.5 score on the development set by following the instructions to run [inference](##inference).

### Merge feed-forward weights

You can reduce computational costs by merging the backbone transformer feed-forward layer with the experts' feed-forward layer at each transformer block.

```bash
./merge-weights.sh ${path_to_best_checkpoint}
```

The command will merge the weights and tune the merged weights (while freezing other weights) for 30 gradient updates. As before, choose the checkpoint that produces the highest F0.5 score on the development set as the final model.

## License

This repository is licensed under the GNU General Public License Version 3 (see [License](./LICENSE.txt)).