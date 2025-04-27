git clone https://github.com/BoyuanJiang/FitDiT.git

cd FitDiT

apt update && apt install git-lfs -y
git lfs install

git clone https://huggingface.co/BoyuanJiang/FitDiT local_model_dir/

pip install -r requirements.txt (recommend you use uv pip install
