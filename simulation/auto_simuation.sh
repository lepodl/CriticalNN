#!/bin/bash
for nfiles in {0..30}
do
  echo $nfiles
  cd /public/home/ssct004t/project/zenglb/CriticalNN/data/multi_size_block/size_49/single
  fileNum=`ls -l |grep "^-"|wc -l`
  cpus=`expr $fileNum / 4 + 1`
  echo "nodes: $cpus"
  cd /public/home/ssct004t/project/zenglb/CriticalNN/simulation
  server_file="server_dtb.slurm"
  sed -in "10c#SBATCH -N ${cpus}" $server_file
  job_id=`sbatch server_dtb.slurm 2>&1 | tr -cd "[0-9]"`
  echo $job_id
  sleep 20s
  cd log
  job_out_file="${job_id}.o"
  str_row_ip=`cat $job_out_file | grep 'listening' | sed -n 1p`
  ip=`echo $str_row_ip | tr -cd "[0-9][.][:]"`
  echo $ip
  cd ../
  client_file="sim.slurm"
  sed -in "2c#SBATCH -J nn_${nfiles}" $client_file
  sed -in "17c \ \ --ip=${ip} \\\\" $client_file  # modify ip
  sed -in "18c \ \ --block_path=\"\/public\/home\/ssct004t\/project\/zenglb\/CriticalNN\/data\/multi_size_block\/size_49\/single\" \\\\" $client_file  # modify write path
  sed -in "19c \ \ --write_path=\"\/public\/home\/ssct004t\/project\/zenglb\/CriticalNN\/data\/100m_scale_block\/random_${nfiles}\" \\\\" $client_file  # modify write path
  sed -in "20c \ \ --idx=${nfiles} \\\\" $client_file  # modify ip
  sbatch $client_file
  sleep 5s
done 
