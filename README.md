Single View to 3D
========================
**Name: Omkar Chittar**  
------------------------
```
Single_View_to_3D
+-dataset
+-output
+-README.md
+-report
+-requirements.txt
```

# **Installation**

- Download and extract the files.
- Make sure you meet all the requirements given on: https://github.com/848f-3DVision/assignment2/tree/main
- The **dataset** folder consists of all the data necessary for the code.
- Unzip the `r2n2_shapenet_dataset` folder in the **dataset** folder.
- The **output** folder has all the images/gifs generated after running the codes.
- All the necessary instructions for running the code are given in **README.md**.
- The folder **report** has the html file that leads to the webpage.


# **1. Exploring loss functions**
## **1.1 Fitting a voxel grid**
Run the code:  
   ```
 python render_voxel.py
   ``` 
The code renders the ground truth and the optimized voxel grid is saved as *gt_vox_grid.gif* and *opt_vox_grid* respectively in the output folder. 

## **1.2 Fitting a point cloud**
Run the code:  
   ```
 python render_pointcloud.py
   ```
The code renders the ground truth and the optimized voxel grid is saved as *gt_pointcloud.gif* and *opt_pointcloud* respectively in the output folder. 

## **1.3 Fitting a mesh**
Run the code:  
   ```
 python render_mesh.py
   ```
The code renders the ground truth and the optimized voxel grid is saved as *gt_mesh.gif* and *opt_mesh* respectively in the output folder. 

# **2. Reconstructing 3D from single view**
## **2.1 Image to voxel grid**
Training:  
   ```
 python train_model.py --type 'vox'
   ```

followed by  

Evaluation:
   ```
 python eval_model.py --type 'vox' --load_checkpoint 
   ```
If the training stops abruptly due to GPU limitations, execute the command again while including the option '--load_checkpoint'.  
The renders/visualizations are saved in the output folder. 

## **2.2 Image to point cloud**
Training:  
   ```
 python train_model.py --type 'point'
   ```

followed by  

Evaluation:
   ```
 python eval_model.py --type 'point' --load_checkpoint 
   ```
If the training stops abruptly due to GPU limitations, execute the command again while including the option '--load_checkpoint'.  
The renders/visualizations are saved in the output folder. 

## **2.3 Image to mesh**
Training:  
   ```
 python train_model.py --type 'mesh'
   ```

followed by  

Evaluation:
   ```
 python eval_model.py --type 'mesh' --load_checkpoint 
   ```
If the training stops abruptly due to GPU limitations, execute the command again while including the option '--load_checkpoint'.  
The renders/visualizations are saved in the output folder. 

## **2.4 Quantitative comparisions**
Running the code, 
   ```
 python eval_model.py --type voxel|mesh|point --load_checkpoint
   ```
produces evaluation plots for the respective representations and is stored in the parent folder.

## **2.5 Analyse effects of hyperparms variations**
I have tried observing the effects of variation of the argument 'n_points' on the point cloud render, the values chosen being [1000, 5000, 10000] and visualized their respective eval plots.
You can repeat the process by executing the commands for pointcloud as mentioned above with the value of n_points of your choice.

## **2.6 Interpret your model**
To interpret what my model is learning, I tried to visualize how the mesh learns during training.

I stored images of the predicted mesh at different training steps and then combined them into a single gif.

As we see in the gif, the mesh is highly unstable in the beginning and stabilizes over time.

# **3. Webpage**
The html code for the webpage is stored in the *report* folder along with the images/gifs.
Clicking on the *webpage.md.html* file will take you directly to the webpage.


