mkdir Benchmark/Data

# NDS
wget https://dl.fbaipublicfiles.com/nds/data.zip -P Benchmark/Data
unzip Benchmark/Data/data.zip -d Benchmark/Data

# NB101
git clone https://github.com/google-research/nasbench.git Benchmark/Data/nasbench
python3 Benchmark/preparenb101.py
pip install Benchmark/Data/nasbench
pip install protobuf==3.20.0
wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord -P Benchmark/Data
python3 Benchmark/Data/nasbench/nasbench/scripts/generate_graphs.py --output_file Benchmark/Data/nasbench/generated_graphs.json

# NB201
pip install nas-bench-201
gdown https://drive.google.com/uc?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs -O Benchmark/Data/NB201.pth

# Macro
wget https://raw.githubusercontent.com/xiusu/NAS-Bench-Macro/master/data/nas-bench-macro_cifar10.json -P Benchmark/Data

# Fast-Soft-Sort
git clone https://github.com/google-research/fast-soft-sort.git 
# cd fast-soft-sort
# python setup.py install
# cd ..