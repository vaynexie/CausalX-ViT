# ViT-CX: Causal Explanation of Vision Transformers

ViT-CX: a tool for making explanations for the ViT image classification.

Paper: https://arxiv.org/abs/2211.03064


**Requirements**

We include a requirements.txt file for the specific environment we used to run the code. To run the code, please set up your environment to match that.

**Our Testing Environment**
Python 3.9.16
CUDA  11.6
pytorch 1.13.1+cu116
torchvision 0.14.1+cu116

**Usage Example**

See Usage_Example.ipynb for the usage example.

**Main Code**

ViT_CX.py: code for making explanation for an example instance.


Noteworthy points:

1.  The hyperparameters in the main function: ViT_CX
```python

def ViT_CX(model,image,target_layer,target_category=None,distance_threshold=0.1,reshape_function=reshape_function_vit,gpu_batch=50)
'''
2. model: ViT model to be explained;
3. image: input image in the tensor form (shape: [1,#channels,width,height]);
4. target_layer: the layer to extract feature maps  (e.g. model.blocks[-1].norm1);
5. target_category: int (class to be explained), in default - the top_1 prediction class;
6. distance_threshold: float between [0,1], distance threshold to make the clustering where  
   feature maps with pairwise similarity<distance_threshold will be merged together, in default - 0.1; 
7. reshape_function: function to reshape the extracted feature maps, in default - a reshape function for vanilla vit;
8. gpu_batch: batch size the run the prediction for the masked images, in default - 50.
'''
```

2. Hyperparameter: reshape_function 
In ViT-CX, only the patch embeddings are used to make the explanation. The hyperparameter ''reshape_function'' is a function to discard the other token embeddings (like, [CLS] token and [DIST] token in DeiT) and reshape the patch embeddings in 1D into the original 2D position relationship. 

This hyperparameter is in default to be a function for reshaping the embeddings from the vanilla ViT model: 

```python
def reshape_function_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
```
However, it should be changed when the explained model is different. For instance, when explaining the DeiT-distilled where the first token in its embeddings is [CLS] token and the second is the distillation token [DIST]. The ''reshape_function" need to be changed to:
```python
def reshape_function_deit_distilled(tensor, height=14, width=14):
    result = tensor[:, 2:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
```

In general, the output of this reshape function should be the embeddings for the image patches, in the shape of [number of feature maps, sqrt(number of patches), sqrt(number of patches)].

 
