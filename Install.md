# Instration
Install miniconda  
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
bash Miniconda3-latest-Linux-x86_64.sh  
~/miniconda3/bin/conda init bash  
source ~/.bashrc  
***  
Clone two repository
git clone https://github.com/v-iashin/video_features.git
git clone https://github.com/minor0419/generate_tsne.git
cd generate_tsne  
***  
Copy directorys from video_features to generate_tsne
cp -r ../video_features/models .
cp -r ../video_features/utils .
cp -r ../video_features/configs .
***  
Make conda enviroment
conda env create conda_env_torch_zoo.yml  
conda activate torch_zoo  
pip install matplotlab  
***  
Install Tsnecuda for your CUDA version  
# CUDA 11.0  
pip3 install tsnecuda==3.0.1+cu110 -f https://tsnecuda.isx.ai/tsnecuda_stable.html  
# CUDA 11.1  
pip3 install tsnecuda==3.0.1+cu111 -f https://tsnecuda.isx.ai/tsnecuda_stable.html  
# CUDA 11.2  
pip3 install tsnecuda==3.0.1+cu112 -f https://tsnecuda.isx.ai/tsnecuda_stable.html  
# CUDA 11.3  
pip3 install tsnecuda==3.0.1+cu113 -f https://tsnecuda.isx.ai/tsnecuda_stable.html  
# CUDA 10.1  
pip3 install tsnecuda==3.0.1+cu101 -f https://tsnecuda.isx.ai/tsnecuda_stable.html  
# CUDA 10.0  
pip3 install tsnecuda==3.0.1+cu100 -f https://tsnecuda.isx.ai/tsnecuda_stable.html  
***   
cp ./test/* .  
python generate_tsne.py  
