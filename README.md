# NeuroPrim

Implementation of paper:

Shi Yuchen, Han Congying, Guo Tiande, [NeuroPrim: An Attention-based Model for Solving NP-hard Spanning Tree Problems](https://www.sciengine.com/SCM/doi/10.1007/s11425-022-2175-5), SCIENCE CHINA Mathematics, 2023.

![neuroprim](images/neuroprim.png)

## Dependencies

- Python 3.10
- PyTorch 2.0.1
- TensorBoard 2.10.0
- tqdm 4.65.0
- wandb 0.15.4

## Quick Start

### Training

With the default settings of `--num_batches 2500` and `–-num_epochs 100`, the training of each problem with 20 vertices will take about 10 hours on a single RTX 3090 GPU.

```bash
python generate_data.py --graph_size 20
python train --graph_size 20 --degree_constrain 2
python train --graph_size 20 --cost_function_key mrcst
python train --graph_size 20 --is_stp True
```

### Testing

Run `test.ipynb` to see the performance of pretrained models on random data.

## Example Solutions

|        DCMST (d=2)         |           MRCST            |          STP           |
| :------------------------: | :------------------------: | :--------------------: |
| ![dcmst](images/dcmst.png) | ![mrcst](images/mrcst.png) | ![stp](images/stp.png) |

## Acknowledgements

- [wouterkool](https://github.com/wouterkool)/**[attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)**: data generation in `generate_data.py`
- [xbresson](https://github.com/xbresson)/**[TSP_Transformer](https://github.com/xbresson/TSP_Transformer)**: network built in `nets/Bresson2021TheTN.py`
- [Transformer — PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html): Transformer used in `nets/transformer.py`

## Citation

If you find our paper or repository helpful for your research or project, please considering citing our paper with the following BibTeX citation format:

```tex
@article{shi2023neuroprim,
  author = {Shi, Yuchen and Han, Congying and Guo, Tiande},
  title = {{NeuroPrim}: An Attention-based Model for Solving NP-hard Spanning Tree Problems},
  journal = {SCIENCE CHINA Mathematics},
  year = {2023},
  url = {https://www.sciengine.com/SCM/doi/10.1007/s11425-022-2175-5},
  doi = {10.1007/s11425-022-2175-5}
}
```
