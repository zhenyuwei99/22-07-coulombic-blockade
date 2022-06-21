# $1 device: eg 3080
# $2 kernel name
# $3 job label
time=$(date "+%Y-%m-%d")
device=$1
label=$2
job_name=$time-fdpe-$label
parent_dir=./out/$device-nsys-out
dir_name=$parent_dir/$job_name
if [ ! -d $parent_dir  ];then
  mkdir $parent_dir
fi
rm -rf $dir_name
mkdir $dir_name
nsys profile --stats=true -o $dir_name/$job_name python ./benchmark.py
