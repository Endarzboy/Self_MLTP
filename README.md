# Self-Supervised Multi-Label Transformation Prediction for Video Representation Learning
This is the implementation for "**MLTP: Self-Supervised Multi-Label Transformation Prediction for Video Representation Learning**". The paper can be found [here](https://www.worldscientific.com/doi/abs/10.1142/S0218126622501596)

### Proposed Proxy Task

<div style='float: center'>
  <img style='width: 1000px' src="./figures/old_results/mltp.png"></img>
</div>


If you want to run the *MLTP pre-training*, run the following command,
```
python main.py
```
If you want to run the *Inference*, run the following command,
``` 
python inference.py
```
If you want to run the *Fine-tuning*, run the following command,
```
python fine_tune.py
```

### Qualitative Results



## Citation
If you find this code useful for your research, please cite our paper:

    @article{assefa2022self,
      title={Self-Supervised Multi-Label Transformation Prediction for Video Representation Learning},
      author={Assefa, Maregu and Jiang, Wei and Yilma, Getinet and Kumeda, Bulbula and Ayalew, Melese and Seid, Mohammed},
      journal={Journal of Circuits, Systems and Computers},
      pages={2250159},
      year={2022},
      publisher={World Scientific}
    }
