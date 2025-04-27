

# YOLO + Segmentation + Pose Estimation + Clothing Tryon + Remapping

apt update && apt install git-lfs -y

git lfs install

git clone https://huggingface.co/BoyuanJiang/FitDiT local_model_dir/

pip install -r requirements.txt (recommend you use uv pip install)

# 2D ==> 3D

https://github.com/Tencent/Hunyuan3D-2

pip install -r requirements.txt
pip install -e .

cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install

pip install huggingface_hub==0.16.4 transformers==4.30.2
