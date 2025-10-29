#!/bin/bash

if [ $# -lt 2 ]; then
  echo "usage: submit_slurm.sh CMD NCPUS"
  exit 0;
fi

cmd=${1}
ncpus=${2}

if [ $# -eq 3 ]; then
  queue="$3"
else
  queue="cpu"
fi

#rm -rf run.sh
rndstr=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c10)

mkdir -pv slurm_output

cat <<EOT >> run_${rndstr}.sh
#!/bin/bash
#SBATCH -J autosub
#SBATCH -p ${queue}
#SBATCH -N 1
##SBATCH --exclude=cpu040
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --ntasks-per-node=${ncpus}
#SBATCH -n ${ncpus}
#SBATCH --gres=gpu:0
#SBATCH --get-user-env
#SBATCH -e slurm_output/%j.err
#SBATCH -o slurm_output/%j.out

module load gcc/11.2.0
module load openmpi/4.1.2
module list
hostname

date
time ${cmd}

date

echo "COMPLETE ..."
EOT

sbatch run_${rndstr}.sh

rm -rfv run_${rndstr}.sh
