#!/bin/bash
for nfiles in {0..49}
do
  echo $nfiles
  cd /public/home/ssct004t/project/zenglb/CriticalNN/data/mutli_size_block/size_${nfiles}/single
  fileNum=`ls -l |grep "^-"|wc -l`
  cpus = `expr $fileNum / 4 + 1`
  cd /public/home/ssct004t/project/zenglb/CriticalNN/simulation
  server_file = "server_dtb"
  sed -in "10c \ \ --SBATCH -N ${cpus} \\\\" $server_file
  job_id=`sbatch server_dtb.slurm 2>&1 | tr -cd "[0-9]"`
  echo $job_id
  sleep 30s
  cd log
  job_out_file="${job_id}.o"
  str_row_ip=`cat $job_out_file | grep 'listening' | sed -n 1p`
  ip=`echo $str_row_ip | tr -cd "[0-9][.][:]"`
  echo $ip
  client_file="sim.slurm"
  sed -in "2c#SBATCH -J nn_${nfiles}" $client_file
  sed -in "17c \ \ --ip=${ip} \\\\" $client_file  # modify ip
  sed -in "18c \ \ --block_path=\"\/public\/home\/ssct004t\/project\/zenglb\/CriticalNN\/data\/mutli_size_block\/size_${nfiles}\/single\" \\\\" $client_file  # modify write path
  sed -in "19c \ \ --write_path=\"\/public\/home\/ssct004t\/project\/zenglb\/CriticalNN\/data\/mutli_size_result\size_${nfiles}\" \\\\" $client_file  # modify write path
  sbatch $client_file
  sleep 10s
  cd ../
done 
