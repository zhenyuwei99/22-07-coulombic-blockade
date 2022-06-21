# $1 device: eg 3080
# $2 kernel name
# $3 job label
time=$(date "+%Y-%m-%d")
device=$1
name=$2
label=$3
job_name=$time-$label-$name
parent_dir=./out/$device-ncu-out
dir_name=$parent_dir/$job_name
if [ ! -d $parent_dir  ];then
  mkdir $parent_dir
fi
rm -rf $dir_name
mkdir $dir_name
ncu_exe=/home/zhenyuwei/Programs/nvidia/ncu/ncu
python_exe=/home/zhenyuwei/Programs/anaconda3/envs/mdpy-dev/bin/python

sudo $ncu_exe --kernel-regex $name --target-processes all --set full -o $dir_name/$job_name --launch-skip 10 --launch-count 1 "$python_exe" ./benchmark.py
