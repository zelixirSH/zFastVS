#!/bin/bash

if [ $# -lt 2 ]; then
  echo "usage: submit_slurm.sh CMD NCPUS taskid [genmsa]"
  exit 0;
fi

cmd=${1}
ncpus=${2}
taskid=${3}

if [ $# -eq 4 ]; then
  queue="$4"
else
  queue="normal"
fi

# make log files
currdir=`echo $PWD`
mkdir -pv "$currdir/logs_slurm"

#rm -rf run.sh
rndstr=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c10)

cat <<EOT >> run_${rndstr}.sh
#!/bin/bash
#SBATCH -J ${taskid}
#SBATCH -p ${queue}
#SBATCH -N 1
#SBATCH --mem-per-cpu=2GB
#SBATCH --ntasks-per-node=${ncpus}
#SBATCH -n ${ncpus}
#SBATCH --gres=gpu:0
#SBATCH --get-user-env
#SBATCH -e $currdir/logs_slurm/%j.err
#SBATCH -o $currdir/logs_slurm/%j.out
#SBATCH --exclude=cpu040
###SBATCH --nodelist=cpu039

module load gcc/11.2.0 openmpi/4.1.2
#module load openmpi/4.0.4_cuda10.1
export PATH=$PATH:/sugon_store/zhengliangzhen/.conda/envs/sfct/bin
#export AMBERHOME=/share/zdaemon/miniconda3/envs/protein

hostname
date

${cmd}

date

echo "COMPLETE ..."
EOT

out=`sbatch run_${rndstr}.sh | grep "Submitted" | wc -l`
if [ $out -eq 1 ]; then
  echo "[INFO] `date`: submitted job sbatch run_${rndstr}.sh"
fi

n=0
while  [ $out -ne 1 ]; do
  sleep 3s
  echo "Retry Submit Slurm Job: sbatch run_${rndstr}.sh"
  out=`sbatch run_${rndstr}.sh | grep "Submitted" | wc -l`
  if [ $out -eq 1 ]; then
    echo "[INFO] `date`: submitted job sbatch run_${rndstr}.sh"
  fi

  n=`expr $n + 1`
  if [ $n -gt 3 ]; then
    break 
  fi
done

rm -rfv run_${rndstr}.sh
