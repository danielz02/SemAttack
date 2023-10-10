#!/bin/bash

COUNTER=0
for model in "meta-llama/Llama-2-7b-chat-hf" "chavinlo/alpaca-native" "eachadea/legacy-vicuna-13b" "TheBloke/stable-vicuna-13B-HF"
do
    for task in "sst2" "mnli" "mnli-mm" "qnli" "qqp" "rte"
    do
        echo $COUNTER
        cat << EOF | sbatch
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100g
#SBATCH --output="logs/perturb_candidates_${task}.out"
#SBATCH --error="logs/perturb_candidates_${task}.err"
#SBATCH --cpus-per-task=64
#SBATCH --partition=compute
#SBATCH --job-name=GetFC__${task}
#SBATCH --time=24:00:00 # hh:mm:ss for the job
#SBATCH --gpus-per-node=1


module purge
# source /opt/rh/devtoolset-10/enable
# conda init bash
conda activate alpaca
module load mpi/openmpi/4.1.4-gcc

echo "$LD_LIBRARY_PATH"
which mpiexec
which python

cd /data/chenhui_zhang/SemAttack || exit
ifconfig

NODE_LIST=$( scontrol show hostname "$SLURM_JOB_NODELIST" | sed -z 's/\n/\:4,/g' )
NODE_LIST=${NODE_LIST%?}

echo "Job is starting on $(hostname) with $SLURM_NTASKS tasks on $NODE_LIST"
python attacks/get_FC.py --task "${task}" --model "${model}" --word-list "./static/${model}/word_list.npy" --embedding-space "./static/${model}/s.npy" --split "train"
python attacks/get_FT.py --task "${task}" --model "${model}" --word-list "./static/${model}/word_list.npy" --embedding-space "./static/${model}/s.npy" --split "train"
python attacks/get_FK.py --task "${task}" --model "${model}" --word-list "./static/${model}/word_list.npy" --embedding-space "./static/${model}/s.npy" --split "train"
EOF
    COUNTER=$((COUNTER+1))
    done
done